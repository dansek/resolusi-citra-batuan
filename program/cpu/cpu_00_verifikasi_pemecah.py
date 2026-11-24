#!/usr/bin/env python3
"""
program/cpu/cpu_00_verifikasi_pemecah.py
Langkah 00 [CPU]: verifikasi pemecah LBM terhadap solusi Stokes analitik (luaran wajib 5, P9).
Pembungkus tipis untuk program/inti/inti_00_verifikasi_pemecah.py dengan solver CPU.
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.pemuat_modul import muat
_vp = muat("program/inti/inti_00_verifikasi_pemecah.py")

DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil", "cpu")


def jalankan(out_dir=DEFAULT_OUT_DIR, **_):
    return _vp.jalankan(backend="cpu", out_dir=os.path.join(out_dir, "verifikasi"))


def main():
    p = argparse.ArgumentParser(description="Langkah 00 [CPU]: verifikasi pemecah (P9)")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sisi", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--diameter", nargs="+", type=int, default=[8, 16, 32])
    a = p.parse_args()
    _vp.jalankan(backend="cpu", out_dir=os.path.join(a.out_dir, "verifikasi"), sisi=tuple(a.sisi), diameter=tuple(a.diameter))


if __name__ == "__main__":
    main()
