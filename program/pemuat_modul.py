"""
program/pemuat_modul.py
=======================
Pemuat modul berprefiks angka (misal program/inti/inti_02_rentang_segmentasi.py), yang tidak dapat
diimpor dengan pernyataan `import` biasa karena namanya diawali angka.
"""
import os
import importlib.util

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_CACHE = {}


def muat(jalur_relatif):
    """Muat modul dari jalur relatif terhadap folder repositori, misal 'program/inti/inti_02_rentang_segmentasi.py'."""
    jalur = os.path.join(ROOT, jalur_relatif)
    if jalur not in _CACHE:
        nama = "m_" + jalur_relatif.replace("/", "_").replace(".py", "")
        spec = importlib.util.spec_from_file_location(nama, jalur)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[jalur] = mod
    return _CACHE[jalur]
