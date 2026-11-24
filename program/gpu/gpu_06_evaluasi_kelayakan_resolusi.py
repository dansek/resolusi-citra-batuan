#!/usr/bin/env python3
"""
program/gpu/gpu_06_evaluasi_kelayakan_resolusi.py
Langkah 06 [GPU]: evaluasi kelayakan resolusi (Bab 8 langkah 4, 6, 7, 8).
Pembungkus tipis untuk program/inti/inti_06_evaluasi_kelayakan_resolusi.py (hanya membaca hasil Langkah 01-05 backend GPU).
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.pemuat_modul import muat
_ev = muat("program/inti/inti_06_evaluasi_kelayakan_resolusi.py")

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil", "gpu")


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, **_):
    return _ev.jalankan(rocks=rocks, out_dir=out_dir)


def main():
    p = argparse.ArgumentParser(description="Langkah 06 [GPU]: evaluasi kelayakan resolusi")
    p.add_argument("--rocks", nargs="+", default=["Bentheimer", "Parker"])
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    a = p.parse_args()
    jalankan(a.rocks, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
