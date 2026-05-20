"""Load submodules from this repository's ``src/diffusers`` tree without HF ``sys.modules`` collisions."""

import importlib.util
import sys
from pathlib import Path

_LOCAL_ROOT = Path(__file__).resolve().parents[1]


def _ensure_local_packages(relative_path: str) -> None:
    parts = Path(relative_path).parts
    for index in range(len(parts)):
        package_name = "diffusers" if index == 0 else "diffusers." + ".".join(parts[:index])
        if package_name in sys.modules and getattr(sys.modules[package_name], "__mdt_local__", False):
            continue
        if package_name in sys.modules:
            continue
        package_root = _LOCAL_ROOT if index == 0 else _LOCAL_ROOT.joinpath(*parts[:index])
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_root / "__init__.py",
            submodule_search_locations=[str(package_root)],
            
        )
        if spec is None or spec.loader is None:
            module = type(sys)(package_name)
            module.__path__ = [str(package_root)]
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        module.__mdt_local__ = True
        sys.modules[package_name] = module


def load_local_module(relative_path: str):
    """Load a module from this repo, e.g. ``schedulers/scheduling_ddpm_mdt``."""
    module_name = "diffusers." + relative_path.replace("/", ".")
    if module_name in sys.modules and getattr(sys.modules[module_name], "__mdt_local__", False):
        return sys.modules[module_name]

    _ensure_local_packages(relative_path)
    file_path = _LOCAL_ROOT / f"{relative_path}.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
        submodule_search_locations=[str(_LOCAL_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name.rsplit(".", 1)[0]
    module.__mdt_local__ = True
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
