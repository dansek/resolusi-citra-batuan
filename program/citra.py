"""
program/citra.py
================
Satu-satunya pembaca citra biner .raw untuk seluruh langkah (GPU, CPU, inti).
Konvensi berkas: uint8, 0 = pori, 1 = padat, urutan C [z, y, x], 256^3, dx = 2,25 um.
Dalam mode uji (program/mode_uji.py) citra dipotong kubus di tengah.
"""

import numpy as np

from program import mode_uji

N_ASLI = 256


def baca_citra_biner(jalur_raw, n=N_ASLI, urutan="C"):
    v = np.fromfile(jalur_raw, dtype=np.uint8)
    if v.size != n ** 3:
        raise ValueError(f"Ukuran berkas {jalur_raw} ({v.size}) tidak sesuai {n}^3.")
    return mode_uji.potong(v.reshape((n, n, n), order=urutan) == 0)
