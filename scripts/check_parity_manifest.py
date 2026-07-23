#!/usr/bin/env python3
"""Validate pinned parity inventories without third-party dependencies."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "parity"
UMBRELLA = ROOT.parent
VALID = {"mapped", "adapter", "reference", "optimized", "missing", "internal", "non_kernel"}
MATRIX_VALID = {"planned", "reference", "optimized", "not_applicable"}
MATRIX_SCHEMES = {
    "nvfp4", "mxfp4", "mxfp8", "fp8_e4m3", "fp8_e5m2", "fp8_w8a16",
    "int4_w4a16", "int4_w4a4", "int4_w4a8", "int8_w8a16", "int8_w8a8",
    "awq", "gptq", "autoround", "smoothquant", "turboquant",
    "bitnet_b1_58", "bitnet_a4_8",
}
MATRIX_BENCHMARKS = {
    "quant_formats", "quant_activation", "quant_gemv_matrix",
    "quant_gemm_matrix", "quant_fusions", "quant_serving", "quant_kv",
    "bitnet_matrix",
}
CONTRACT_LAYOUTS = {
    "int4_symmetric", "uint4_affine", "int8_symmetric", "int8_affine",
    "fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1", "mxfp8_e4m3_e8m0",
    "mxfp4_e2m1_e8m0", "nvfp4_e2m1_e4m3", "bitnet_ternary",
    "turboquant_key", "turboquant_value",
}
IMPORT_GOLDENS = {
    "awq_u4", "gptq_u4", "autoround_u4", "autoround_fp8",
    "autoround_mxfp4", "autoround_mxfp8", "autoround_nvfp4",
    "smoothquant_w8a8", "bitnet_i2_s",
}


def rows(name: str) -> list[dict[str, str]]:
    with (PARITY / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fail(message: str) -> None:
    print(f"parity manifest: {message}", file=sys.stderr)
    raise SystemExit(1)


def git_head(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_sources() -> tuple[Path, dict[str, Path]]:
    source_rows = rows("sources.tsv")
    if len(source_rows) != 5:
        fail(f"expected five pinned sources, found {len(source_rows)}")
    llama = None
    source_paths: dict[str, Path] = {}
    for row in source_rows:
        path = Path(row["path"])
        if not path.is_dir():
            fail(f"missing checkout for {row['source']}: {path}")
        actual = git_head(str(path))
        if not actual.startswith(row["revision"]):
            fail(f"{row['source']} is {actual}, manifest pins {row['revision']}")
        if row["source"] == "llama.cpp":
            llama = path
        else:
            source_paths[row["source"]] = path
    if llama is None:
        fail("llama.cpp source row is missing")
    return llama, source_paths


def validate_llama_ops(llama: Path) -> None:
    cpu = llama / "ggml" / "src" / "ggml-cpu"
    pattern = re.compile(r"GGML_OP_[A-Z0-9_]+")
    discovered: set[str] = set()
    for path in cpu.rglob("*"):
        if path.suffix not in {".c", ".cc", ".cpp", ".h", ".hpp"}:
            continue
        discovered.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
    manifest_rows = rows("llama_ops.tsv")
    manifest = {row["source_symbol"] for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        fail("duplicate llama operation row")
    if discovered != manifest:
        fail(
            "llama CPU op drift: missing="
            f"{sorted(discovered - manifest)}, stale={sorted(manifest - discovered)}"
        )
    for row in manifest_rows:
        if row["status"] not in VALID:
            fail(f"invalid llama op status {row['status']} for {row['source_symbol']}")
    missing = [row["source_symbol"] for row in manifest_rows if row["status"] == "missing"]
    if missing:
        fail(f"llama operation parity is incomplete: {missing}")

    validate_mapped_symbols(manifest_rows)


def validate_mapped_symbols(manifest_rows: list[dict[str, str]]) -> None:
    public_headers = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "include" / "quixicore_cpu").glob("*.h")
    )
    for row in manifest_rows:
        if row["status"] != "mapped":
            continue
        for symbol in row["cpu_mapping"].split("/"):
            if not re.search(rf"\b{re.escape(symbol)}\s*\(", public_headers):
                fail(f"mapped CPU symbol is not public: {symbol} ({row['source_symbol']})")


def validate_nested_enum(llama: Path, enum_name: str, prefix: str,
                         manifest_name: str) -> None:
    header = (llama / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
    header = re.sub(r"//[^\n]*", "", header)
    enum_match = re.search(rf"enum\s+{re.escape(enum_name)}\s*\{{(.*?)\}};", header, re.S)
    if enum_match is None:
        fail(f"cannot locate enum {enum_name}")
    discovered = set(re.findall(rf"{re.escape(prefix)}[A-Z0-9_]+", enum_match.group(1)))
    manifest_rows = rows(manifest_name)
    manifest = {row["source_symbol"] for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        fail(f"duplicate row in {manifest_name}")
    if discovered != manifest:
        fail(
            f"{enum_name} drift: missing={sorted(discovered - manifest)}, "
            f"stale={sorted(manifest - discovered)}"
        )
    for row in manifest_rows:
        if row["status"] not in VALID:
            fail(f"invalid status {row['status']} for {row['source_symbol']}")
    missing = [row["source_symbol"] for row in manifest_rows if row["status"] == "missing"]
    if missing:
        fail(f"{enum_name} parity is incomplete: {missing}")
    validate_mapped_symbols(manifest_rows)


def top_level_map_names(path: Path, section: str) -> set[str]:
    names: set[str] = set()
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == f"{section}:":
            in_section = True
            continue
        if in_section and line and not line.startswith(" "):
            break
        if in_section:
            match = re.fullmatch(r"  ([A-Za-z0-9_]+):\s*", line)
            if match:
                names.add(match.group(1))
    return names


def validate_umbrella_quant_contract() -> None:
    registry = UMBRELLA / "registry" / "quant-formats.yaml"
    required_families = {
        "gguf", "mx", "fp8", "fp4", "bitnet", "integer",
        "checkpoint_schemes", "turboquant",
    }
    families = top_level_map_names(registry, "quant_formats")
    if not required_families <= families:
        fail(f"umbrella quant registry is missing: {sorted(required_families - families)}")
    required_tokens = {
        "specs/formats/fp8.md": ("E4M3FN", "E5M2", "round-to-nearest"),
        "specs/formats/fp4.md": ("E2M1", "low nibble", "FP32"),
        "specs/formats/mx-formats.md": ("E8M0", "MXFP8", "MXFP4", "NVFP4"),
        "specs/formats/bitnet.md": ("b1.58", "a4.8", "Canonical ternary"),
        "specs/formats/integer.md": ("Signed INT4", "Affine U4", "Signed INT8"),
        "specs/formats/schemes.md": ("AWQ", "GPTQ", "AutoRound", "SmoothQuant"),
        "specs/formats/turboquant.md": ("least-significant-bit-first", "FWHT", "centroids"),
        "specs/kernels/quantization.md": ("W8A16", "W4A8", "W4A4", "FP32"),
    }
    for relative, tokens in required_tokens.items():
        path = UMBRELLA / relative
        if not path.is_file():
            fail(f"missing umbrella quant contract: {relative}")
        contents = path.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in contents]
        if missing:
            fail(f"umbrella quant contract {relative} lacks {missing}")


def validate_sibling_operations(source_paths: dict[str, Path]) -> None:
    discovered: set[tuple[str, str]] = set()
    for source, path in source_paths.items():
        manifest = path / ".quixicore" / "kernels.yaml"
        if not manifest.is_file():
            fail(f"missing sibling kernel manifest: {manifest}")
        discovered.update(
            (source, name) for name in top_level_map_names(manifest, "operations")
        )
    manifest_rows = rows("sibling_operations.tsv")
    inventory = {(row["source"], row["source_symbol"]) for row in manifest_rows}
    if len(inventory) != len(manifest_rows):
        fail("duplicate sibling operation row")
    if discovered != inventory:
        fail(
            "sibling operation drift: missing="
            f"{sorted(discovered - inventory)}, stale={sorted(inventory - discovered)}"
        )
    for row in manifest_rows:
        if row["status"] not in VALID:
            fail(f"invalid sibling operation status: {row}")
    validate_mapped_symbols(manifest_rows)


def validate_sibling_quant_families(source_paths: dict[str, Path]) -> None:
    discovered: set[tuple[str, str]] = set()
    for source, path in source_paths.items():
        manifest = path / ".quixicore" / "quant-formats.yaml"
        if not manifest.is_file():
            fail(f"missing sibling quant manifest: {manifest}")
        discovered.update(
            (source, name) for name in top_level_map_names(manifest, "formats")
        )
    manifest_rows = rows("sibling_quant_families.tsv")
    inventory = {(row["source"], row["format"]) for row in manifest_rows}
    if len(inventory) != len(manifest_rows):
        fail("duplicate sibling quant-family row")
    if discovered != inventory:
        fail(
            "sibling quant-family drift: missing="
            f"{sorted(discovered - inventory)}, stale={sorted(inventory - discovered)}"
        )
    for row in manifest_rows:
        if row["status"] not in VALID:
            fail(f"invalid sibling quant-family status: {row}")
        row["source_symbol"] = row["format"]
    validate_mapped_symbols(manifest_rows)


def validate_llama_quants(llama: Path) -> None:
    header = (llama / "ggml" / "include" / "ggml.h").read_text(encoding="utf-8")
    header = re.sub(r"//[^\n]*", "", header)
    enum_match = re.search(r"enum\s+ggml_type\s*\{(.*?)\};", header, re.S)
    if enum_match is None:
        fail("cannot locate enum ggml_type")
    live = {
        match.group(1)
        for match in re.finditer(
            r"GGML_TYPE_((?:Q\d|IQ|TQ|MXFP|NVFP)[A-Z0-9_]*)\s*=",
            enum_match.group(1),
        )
    }
    manifest_rows = rows("llama_quants.tsv")
    manifest = {row["format"] for row in manifest_rows}
    if live != manifest:
        fail(
            "llama quant type drift: missing="
            f"{sorted(live - manifest)}, stale={sorted(manifest - live)}"
        )
    if any(row[column] == "missing" for row in manifest_rows
           for column in ("pack", "unpack", "gemv_f32")):
        fail("llama quant pack/unpack/f32-GEMV parity is incomplete")


def validate_status_table(name: str, status_columns: list[str]) -> None:
    table = rows(name)
    for row in table:
        for column in status_columns:
            if row[column] not in VALID:
                fail(f"invalid {column}={row[column]} in {name}: {row}")


def validate_dtype_coverage() -> None:
    table = rows("dtype_coverage.tsv")
    expected = {
        "all_public_float_tensor_arguments",
        "arbitrary_cpp_kernel_composition",
        "activation_softmax_norm_dense_attention",
        "quantized_projection_activations_outputs",
        "portable_conversion",
        "aarch64_conversion",
        "x86_64_conversion",
    }
    scopes = {row["scope"] for row in table}
    if scopes != expected or len(scopes) != len(table):
        fail(f"dtype coverage scope drift: {sorted(scopes ^ expected)}")
    validate_status_table("dtype_coverage.tsv", ["fp16", "bf16"])
    public_headers = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "include" / "quixicore_cpu").glob("*.h")
    )
    for row in table:
        for symbol in row["route"].split("/"):
            if not re.search(rf"\b{re.escape(symbol)}\s*\(", public_headers):
                fail(f"dtype route is not public: {symbol} ({row['scope']})")
    generic = next(
        row for row in table if row["scope"] == "all_public_float_tensor_arguments"
    )
    if generic["fp16"] != "adapter" or generic["bf16"] != "adapter" or generic["accumulator"] != "f32":
        fail("universal storage route must remain FP16/BF16 adapter + FP32 accumulation")


def validate_full_quant_matrix() -> None:
    table = rows("full_quant_matrix.tsv")
    expected_columns = {
        "scheme", "canonical_layout", "activation", "operation",
        "input_storage", "output_storage", "work_packages", "status",
        "evidence", "benchmark_family",
    }
    if not table or set(table[0]) != expected_columns:
        fail("full quant matrix schema drift")
    schemes = {row["scheme"] for row in table}
    if schemes != MATRIX_SCHEMES:
        fail(f"full quant scheme drift: {sorted(schemes ^ MATRIX_SCHEMES)}")
    keys: set[tuple[str, str, str, str]] = set()
    for row in table:
        key = (row["scheme"], row["activation"], row["operation"], row["canonical_layout"])
        if key in keys:
            fail(f"duplicate full quant matrix row: {key}")
        keys.add(key)
        if row["status"] not in MATRIX_VALID:
            fail(f"invalid full quant status: {row}")
        if row["benchmark_family"] not in MATRIX_BENCHMARKS:
            fail(f"invalid full quant benchmark family: {row}")
        layouts = set(row["canonical_layout"].split("|"))
        if "target_dependent" not in layouts and not layouts <= CONTRACT_LAYOUTS:
            fail(f"unknown canonical layout in full quant matrix: {row}")
        packages = row["work_packages"].split(",")
        if any(re.fullmatch(r"[CLQKFSAB][0-9]+", package) is None
               for package in packages):
            fail(f"invalid work package in full quant matrix: {row}")
        if row["status"] in {"reference", "optimized"}:
            if row["evidence"] == "-":
                fail(f"quant support claim lacks evidence: {row}")
            for item in row["evidence"].split(","):
                if not (ROOT / item).exists():
                    fail(f"quant evidence path does not exist: {item}")
    lifecycle = {
        row["scheme"] for row in table
        if row["operation"] in {
            "canonical_pack", "checkpoint_import", "cache_encode_decode", "format_reuse"
        }
    }
    if lifecycle != MATRIX_SCHEMES:
        fail(f"schemes without lifecycle/import row: {sorted(MATRIX_SCHEMES - lifecycle)}")
    registry = (ROOT / "benchmarks" / "harness" / "registry.cpp").read_text(
        encoding="utf-8"
    )
    registered = set(re.findall(r'\{"([a-z0-9_]+)",\s*&build_', registry))
    if not MATRIX_BENCHMARKS <= registered:
        fail(f"unregistered full quant benchmark families: {sorted(MATRIX_BENCHMARKS - registered)}")


def validate_quant_goldens() -> None:
    contract_path = ROOT / "tests" / "testdata" / "quant_contract_golden.tsv"
    with contract_path.open(encoding="utf-8", newline="") as handle:
        contract = list(csv.DictReader(handle, delimiter="\t"))
    layouts = {row["layout"] for row in contract}
    if layouts != CONTRACT_LAYOUTS or len(layouts) != len(contract):
        fail(f"quant contract golden drift: {sorted(layouts ^ CONTRACT_LAYOUTS)}")
    for row in contract:
        for column in ("element_bytes_hex", "scale_bytes_hex"):
            value = row[column]
            if value != "-" and (len(value) % 2 != 0 or re.fullmatch(r"[0-9a-f]+", value) is None):
                fail(f"invalid golden hex in {column}: {row}")

    import_path = ROOT / "tests" / "testdata" / "quant_import_golden.tsv"
    with import_path.open(encoding="utf-8", newline="") as handle:
        imports = list(csv.DictReader(handle, delimiter="\t"))
    importers = {row["importer"] for row in imports}
    if importers != IMPORT_GOLDENS:
        fail(f"quant importer golden drift: {sorted(importers ^ IMPORT_GOLDENS)}")
    keys = {(row["importer"], row["case"]) for row in imports}
    if len(keys) != len(imports):
        fail("duplicate quant importer golden")
    for row in imports:
        if row["canonical_layout"] not in CONTRACT_LAYOUTS:
            fail(f"importer golden has unknown canonical layout: {row}")
        value = row["canonical_codes_hex"]
        if len(value) % 2 != 0 or re.fullmatch(r"[0-9a-f]+", value) is None:
            fail(f"invalid importer golden hex: {row}")


def main() -> int:
    validate_umbrella_quant_contract()
    llama, source_paths = validate_sources()
    validate_llama_ops(llama)
    validate_nested_enum(
        llama, "ggml_unary_op", "GGML_UNARY_OP_", "llama_unary_ops.tsv"
    )
    validate_nested_enum(
        llama, "ggml_glu_op", "GGML_GLU_OP_", "llama_glu_ops.tsv"
    )
    validate_sibling_operations(source_paths)
    validate_sibling_quant_families(source_paths)
    validate_llama_quants(llama)
    validate_status_table(
        "llama_quants.tsv",
        ["pack", "unpack", "gemv_f32", "activation_dot", "qgemm", "neon", "avx2", "avx512"],
    )
    validate_status_table("sibling_semantics.tsv", ["status"])
    validate_dtype_coverage()
    validate_full_quant_matrix()
    validate_quant_goldens()
    quant_rows = rows("llama_quants.tsv")
    if sum(row["role"] == "stored" for row in quant_rows) != 25:
        fail("stored llama quant inventory must contain exactly 25 rows")
    if {row["format"] for row in quant_rows if row["role"] == "activation"} != {"Q8_1", "Q8_K"}:
        fail("activation quant inventory must be Q8_1 and Q8_K")
    print(
        f"parity manifest ok: {len(rows('llama_ops.tsv'))} llama ops, "
        f"{len(rows('llama_unary_ops.tsv')) - 1} unary modes, "
        f"{len(rows('llama_glu_ops.tsv')) - 1} GLU modes, "
        f"{len(quant_rows)} llama quant types, "
        f"{len(rows('sibling_operations.tsv'))} sibling manifest operations, "
        f"{len(rows('sibling_quant_families.tsv'))} sibling quant families, "
        f"{len(rows('sibling_semantics.tsv'))} sibling semantic areas, "
        f"{len(rows('dtype_coverage.tsv'))} dtype coverage scopes"
        f", {len(rows('full_quant_matrix.tsv'))} full quant matrix cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
