"""tools/_paths.py - where the tools find llama.cpp: its gguf-py (the MTP packer, the fixtures) and the whole tree
(the nvfp4 converter, which runs llama.cpp's own convert_hf_to_gguf.py).

Order: $STRATA_GGUF_PY, then <engine>/vendor/llama.cpp/gguf-py (the copy this backend tracks), then
<engine>/third_party/llama.cpp/gguf-py (what setup.py clones), then the development tree's
../../.ref/llama.cpp/gguf-py.

This module is the same file in tools/ (the repo root, where the backend's vendor copy is reached through
backends/qwen35_llama/) and in backends/qwen35_llama/tools/: the candidate list covers both positions, so the
copies stay byte-identical and one resolver serves every tool.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def _roots() -> list[Path]:
    """The llama.cpp checkouts to try, in order: the one this backend vendors (from the repo root it is
    backends/qwen35_llama/vendor/llama.cpp), the one setup.py clones, then the development tree's .ref."""
    return [ENGINE / "vendor" / "llama.cpp",
            ENGINE / "backends" / "qwen35_llama" / "vendor" / "llama.cpp",
            ENGINE / "third_party" / "llama.cpp",
            ENGINE.parents[min(1, len(ENGINE.parents) - 1)] / ".ref" / "llama.cpp"]   # the dev tree; a shallow install (G:\Strata) has no grandparent


def gguf_py() -> str:
    cands = [os.environ.get("STRATA_GGUF_PY")] + [r / "gguf-py" for r in _roots()]
    for c in cands:
        if c and (Path(c) / "gguf").is_dir():
            return str(c)
    sys.exit("llama.cpp's gguf-py was not found: run setup.py (it clones llama.cpp into third_party/), set "
             "STRATA_GGUF_PY, or use the backend's vendor/llama.cpp/gguf-py; searched: "
             + ", ".join(str(c) for c in cands if c))


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)


def llama_cpp() -> str:
    """The llama.cpp checkout itself, for a tool that needs more than gguf-py: the nvfp4 converter runs llama.cpp's
    convert_hf_to_gguf.py and imports its `conversion` package, so all of it must come from one tree. A bare
    gguf-py (a pip install, a loose unpack) does not qualify, which also keeps a mixed pair - one tree's gguf with
    another tree's conversion - from being assembled behind the caller's back: $STRATA_GGUF_PY takes part here
    only when it points into a full checkout."""
    cands = [Path(gguf_py()).parent] + _roots()
    for c in cands:
        if ((c / "gguf-py" / "gguf").is_dir() and (c / "conversion" / "__init__.py").is_file()
                and (c / "convert_hf_to_gguf.py").is_file()):
            return str(c)
    sys.exit("llama.cpp's converter (a checkout holding convert_hf_to_gguf.py, conversion/ and gguf-py/) was not "
             "found: the nvfp4 tools run llama.cpp's own converter, so they need the whole tree - run setup.py "
             "first, or point STRATA_GGUF_PY at such a checkout's gguf-py; searched: "
             + ", ".join(str(c) for c in cands))


def add_llama_cpp() -> None:
    """sys.path for a tool that needs both: the tree (for `conversion`) and its gguf-py (for `gguf`)."""
    root = llama_cpp()
    for p in (str(Path(root) / "gguf-py"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
