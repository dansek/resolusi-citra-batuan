#!/usr/bin/env python3
"""
program/inti/inti_03_leher_penentu.py
=====================================
Langkah 03 (Bab 8 langkah 2): leher pori PENENTU (pengendali) per arah aliran.

Takrif yang dipakai sebagai d_t utama mengikuti notebook kuliah:
  * P16 `leher`        : "jari-jari leher penentu r_c (erosi terbesar yang ruang porinya masih menembus)";
                         r dinaikkan 0,25 voxel selama {EDT >= r + 0,25} masih menembus.
  * P11 `leher_minimum`: "lebar leher tersempit pada jalur yang menembus".
  * Bab 8 langkah 2 / P2: yang menentukan adalah leher yang TERSEMPIT di sepanjang lintasan.
Perbedaan dari P16: citra batuan tidak periodik, sehingga EDT dihitung tanpa pengubinan (P16 mengubin
citra sintetik periodik 3x3x3), dan ketersambungan diuji antara muka masuk dan keluar tiap arah
dengan ketetanggaan 6 (P4).

d_c = 2 r_c dx. d_t utama = nilai terkecil dari ketiga arah (leher paling membatasi, konservatif);
nilai per arah ikut dilaporkan karena menjelaskan anisotropi permeabilitas.
"""

import numpy as np
from scipy import ndimage as ndi

SUMBU = {"x": 2, "y": 1, "z": 0}          # C-order [z, y, x]
_S6 = ndi.generate_binary_structure(3, 1)


def _menembus(lab, sumbu):
    a = set(np.unique(np.take(lab, 0, axis=sumbu))) - {0}
    b = set(np.unique(np.take(lab, -1, axis=sumbu))) - {0}
    return bool(a & b)


def leher_penentu(pori, dx_um, edt=None, langkah=0.25):
    """Jari-jari leher penentu r_c (voxel) per arah, takrif P16. Satu pelabelan per r untuk ketiga arah."""
    D = ndi.distance_transform_edt(pori) if edt is None else edt
    r_c = {a: None for a in SUMBU}
    r = 0.5
    while any(v is None for v in r_c.values()):
        lab, n = ndi.label(D >= r + langkah, structure=_S6)
        for a, s in SUMBU.items():
            if r_c[a] is None and (n == 0 or not _menembus(lab, s)):
                r_c[a] = r                      # r terakhir yang masih menembus (P16)
        r += langkah
    d_c = {a: 2.0 * r_c[a] * dx_um for a in SUMBU}
    return {
        "takrif": "P16/P11: erosi terbesar yang ruang porinya masih menembus (kenaikan 0,25 voxel)",
        "r_c_voxel": r_c,
        "d_c_um": d_c,
        "d_t_um": min(d_c.values()),
        "arah_pembatas": min(d_c, key=d_c.get),
    }
