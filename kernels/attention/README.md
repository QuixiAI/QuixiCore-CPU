# Attention

Contract families: `causal_attention`, `paged_attention`, `mla_decode`. Spec:
umbrella `specs/kernels/attention.md`. CPU work here is decode-first:
small-batch attention and KV-cache-friendly paths before any prefill
ambition. Status: planned; no implementation yet.
