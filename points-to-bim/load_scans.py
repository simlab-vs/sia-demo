"""Convert the per-element E57 scans into compressed numpy caches.

File names differ slightly between campaigns (``Tab_AB`` vs ``Tab_A_B``,
``Tab_Est-A`` vs ``Tab_Est_A``), so they are normalised to a common element key.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pye57

DATA_DIR = Path("data")
CACHE_DIR = Path("build/cache")


def element_key(filename: str) -> str:
    key = re.sub(r"_5cm_\d{6}_KM$", "", Path(filename).stem).replace("-", "_")
    return re.sub(r"^Tab_([A-H])([A-H])$", r"Tab_\1_\2", key)


def read_e57(path: Path) -> dict[str, np.ndarray]:
    scan = pye57.E57(str(path)).read_scan(0, ignore_missing_fields=True, colors=True, intensity=True)
    xyz = np.c_[scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]]
    rgb = np.c_[scan["colorRed"], scan["colorGreen"], scan["colorBlue"]].astype(np.uint8)
    return {"xyz": xyz, "rgb": rgb, "intensity": scan["intensity"].astype(np.float32)}


def load_element(campaign: str, key: str) -> np.ndarray:
    return np.load(CACHE_DIR / campaign / f"{key}.npz")["xyz"]


def load_elements(campaign: str, prefix: str) -> np.ndarray:
    files = sorted((CACHE_DIR / campaign).glob(f"{prefix}*.npz"))
    return np.concatenate([np.load(f)["xyz"] for f in files])


def campaigns() -> list[str]:
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


def main() -> None:
    for campaign in campaigns():
        out_dir = CACHE_DIR / campaign
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted((DATA_DIR / campaign).glob("*.e57")):
            out = out_dir / f"{element_key(f.name)}.npz"
            if out.exists():
                continue
            np.savez(out, **read_e57(f))
            print(f"cached {f} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
