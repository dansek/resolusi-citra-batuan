#!/usr/bin/env python3
"""
program/cpu/cpu_08_komparasi_cpu_gpu.py
Langkah 08: komparasi GPU vs CPU dari durasi produksi (perangkat menganggur, cacah langkah sama).
Pembungkus tipis untuk program/inti/inti_08_komparasi_cpu_gpu.py; hasilnya sama dari backend mana pun dipanggil.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.pemuat_modul import muat
_k = muat("program/inti/inti_08_komparasi_cpu_gpu.py")


def jalankan(rocks=None, out_dir=None, **_):
    return _k.jalankan(rocks=rocks, out_dir=out_dir)


if __name__ == "__main__":
    _k.main()
