PRESET ?= dev

.PHONY: all dev test perf sanitize macos-x86-test parity configure build run-tests \
	check-layout list-presets clean

all: dev

dev: PRESET = dev
dev: build

test: PRESET = test
test: run-tests

perf: PRESET = perf
perf: run-tests
	@echo "Benchmark with: scripts/bench --preset quick"

sanitize: PRESET = sanitize
sanitize: run-tests

macos-x86-test: PRESET = macos-x86-test
macos-x86-test: run-tests

parity:
	python3 scripts/check_parity_manifest.py

configure: check-layout
	cmake --preset "$(PRESET)"

build: configure
	cmake --build --preset "$(PRESET)"

run-tests: build
	ctest --preset "$(PRESET)"

check-layout:
	@./scripts/check_build_layout.sh

list-presets:
	cmake --list-presets=all

clean:
	cmake --build --preset "$(PRESET)" --target clean
