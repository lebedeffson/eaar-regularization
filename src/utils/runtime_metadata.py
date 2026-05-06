from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _total_ram_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except Exception:
        return None
    return None


def _cpu_model() -> str | None:
    val = platform.processor()
    if val:
        return val
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            return None
    return None


def collect_runtime_metadata(device=None) -> dict:
    if torch is not None:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        torch_version = torch.__version__
    else:
        dev = device or "cpu"
        cuda_available = False
        cuda_version = None
        torch_version = None
    gpu_name = None
    if cuda_available and torch is not None:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "device_used": str(dev),
        "cpu_model": _cpu_model(),
        "gpu_model": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ram_total_bytes": _total_ram_bytes(),
    }
