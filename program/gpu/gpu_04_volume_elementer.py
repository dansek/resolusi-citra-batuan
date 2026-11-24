#!/usr/bin/env python3
"""
program/gpu/gpu_04_volume_elementer.py
Langkah 04 [GPU]: volume elementer representatif, uji ukuran bertingkat + CV (luaran wajib 3, P4).
Pembungkus tipis untuk program/inti/inti_04_volume_elementer.py dengan solver GPU.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.pemuat_modul import muat
_rev = muat("program/inti/inti_04_volume_elementer.py")

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil", "gpu")


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, periksa=500, **_):
    return _rev.jalankan(rocks=rocks, data_dir=data_dir, out_dir=out_dir, backend="gpu", periksa=periksa)


def main():
    p = argparse.ArgumentParser(description="Langkah 04 [GPU]: volume elementer representatif")
    p.add_argument("--rocks", nargs="+", default=["Bentheimer", "Parker"])
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    a = p.parse_args()
    jalankan(a.rocks, a.data_dir, a.out_dir)


if __name__ == "__main__":
    main()
