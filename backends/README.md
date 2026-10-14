# backends/

Standalone research backends that live inside this repository but are **not** part of the official
top-level build. Each backend is its own CMake project with its own build directory and executable.

| backend | what | doc |
|---|---|---|
| `qwen35_llama/` | Imported Qwen3.6/Qwen3.8-Flash MTP research engine (mini Strata v0.1.41 tree + live patched `vendor/llama.cpp`). EXPERIMENTAL; not production. | [docs/backends/qwen35_llama.md](../docs/backends/qwen35_llama.md) |

Rules that apply to every backend here:

- The official top-level build is untouched: no backend is added with `add_subdirectory`, and backend target
  names may collide with official ones (`strata`, `strata_spec`, ...) by design because they are never in the
  same CMake tree.
- Build each backend separately (each has its own build script/doc); do not merge their binaries into the
  official engine.
- These backends carry research experiments; treat EXPERIMENTAL/REJECTED gates as default-off and do not
  promote them.
