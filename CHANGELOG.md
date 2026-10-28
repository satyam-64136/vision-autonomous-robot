# Changelog

## 1.0 · Hack in Hills '26 submission

- Persistent room map with object memory and temporary on-robot storage (`--store`, `--reset-map`)
- Systematic exploration that stops only when nothing reachable is left unsearched
- Theta\* any-angle planning with a safety-margin ladder
- Pure-pursuit path following; local avoidance only when blocked; automatic replanning
- Instant target switching from the saved map; direct approach to a standoff point
- Asynchronous detector with pose-stamped results; `--latency` emulation for benchmarks
- Label-free camera hazard layer for unfamiliar obstacles (`--novel`, `--no-hazard`)
- Regression tests (`test_nav.py`): exploration, detours, repeated target switching
- Docs: architecture, algorithms, VLM design (proposed), hardware plan (proposed), testing
