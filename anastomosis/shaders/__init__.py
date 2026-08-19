"""WGSL loading with a minimal preprocessor.

WGSL has no include mechanism, so this resolves two directives:

* ``//!include name.wgsl`` -- inline another shader file (once per module).
* ``//!struct Name`` -- inline a struct definition generated from the field
  lists in :mod:`anastomosis.gpu_params`, so host-side packing and shader-side
  layout cannot drift apart.
"""

from __future__ import annotations

import functools
from pathlib import Path

from .. import gpu_params

SHADER_DIR = Path(__file__).parent

_STRUCTS = {
    "SimParams": gpu_params.SIM_FIELDS,
    "RenderParams": gpu_params.RENDER_FIELDS,
    "LayerData": gpu_params.LAYER_FIELDS,
    "Event": gpu_params.EVENT_FIELDS,
    "Stats": gpu_params.STATS_FIELDS,
}

# Structs that are not host-packed and so are not derived from a field list.
_RAW_STRUCTS = {"Partial": gpu_params.PARTIAL_WGSL}


def _expand(name: str, seen: set[str]) -> str:
    path = SHADER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"shader {name!r} not found in {SHADER_DIR}")

    out: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//!include"):
            target = stripped.split(maxsplit=1)[1].strip()
            if target in seen:
                continue  # include guard
            seen.add(target)
            out.append(f"// ---- begin {target} ----")
            out.append(_expand(target, seen))
            out.append(f"// ---- end {target} ----")
        elif stripped.startswith("//!struct"):
            struct_name = stripped.split(maxsplit=1)[1].strip()
            if struct_name in _RAW_STRUCTS:
                out.append(_RAW_STRUCTS[struct_name])
            elif struct_name in _STRUCTS:
                out.append(gpu_params.wgsl_struct(struct_name, _STRUCTS[struct_name]))
            else:
                raise KeyError(
                    f"{name}:{lineno}: unknown generated struct {struct_name!r}"
                )
        else:
            out.append(line)
    return "\n".join(out)


@functools.lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the fully expanded source for a shader module."""
    return _expand(name, seen={name})


def all_shader_names() -> list[str]:
    """Every top-level shader, i.e. excluding include-only fragments."""
    return sorted(
        p.name
        for p in SHADER_DIR.glob("*.wgsl")
        if p.name not in {"common.wgsl", "common3d.wgsl", "bindings.wgsl"}
    )
