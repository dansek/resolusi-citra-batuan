#!/usr/bin/env python3
"""
program/cpu/cpu_02_rentang_segmentasi.py
Langkah 02 [CPU]: rentang segmentasi pada resolusi penuh (Bab 8 langkah 1).

Pembungkus tipis untuk program/inti/inti_02_rentang_segmentasi.py: geseran antarmuka setara (Pedoman 5.2)
pada 7 nilai -0,5 ... +0,5 voxel, lalu LBM D3Q19 tiga arah pada tiap ambang dengan solver CPU.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.pemuat_modul import muat
_rs = muat("program/inti/inti_02_rentang_segmentasi.py")

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil", "cpu")


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, periksa=500, **_):
    return _rs.jalankan(rocks=rocks, data_dir=data_dir, out_dir=out_dir, backend="cpu", periksa=periksa)


def main():
    p = argparse.ArgumentParser(description="Langkah 02 [CPU]: rentang segmentasi resolusi penuh")
    p.add_argument("--rocks", nargs="+", default=["Bentheimer", "Parker"])
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--periksa", type=int, default=500)
    a = p.parse_args()
    jalankan(a.rocks, a.data_dir, a.out_dir, a.periksa)


if __name__ == "__main__":
    main()
