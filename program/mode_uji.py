"""
program/mode_uji.py
===================
Mode uji cepat untuk fase perbaikan program (bukan untuk hasil laporan).

Aktif bila variabel lingkungan S4_UJI berisi sisi potongan (mis. "128"); `run.py --uji` mengaturnya.
Dalam mode uji:
  * setiap citra 256^3 dipotong kubus di tengah ke S4_UJI^3 (`potong`),
  * setiap simulasi LBM dibatasi S4_UJI_LANGKAH langkah (bawaan 2.000) dan diperiksa tiap 100 langkah,
  * setiap berkas kasus diberi tanda "mode": "uji"; `katalog_batuan.hasil_sah` menolak berkas uji
    di luar mode uji dan menolak berkas produksi di dalam mode uji, sehingga keduanya tidak tercampur.
Jalur kode yang dipakai sama persis dengan produksi; hanya ukuran dan batas langkah yang berbeda.
"""

import os


def ukuran():
    v = os.environ.get("S4_UJI", "").strip()
    return int(v) if v else None


def aktif():
    return ukuran() is not None


def batas_langkah():
    return int(os.environ.get("S4_UJI_LANGKAH", "2000"))


def potong(pori):
    """Potongan kubus tengah (hanya dalam mode uji)."""
    n = ukuran()
    if n is None or pori.shape[0] <= n:
        return pori
    a = (pori.shape[0] - n) // 2
    return pori[a:a + n, a:a + n, a:a + n].copy()


def atur_lbm(maks_langkah, periksa):
    """Batas langkah dan selang periksa yang dipakai solver."""
    if not aktif():
        return maks_langkah, periksa
    return min(int(maks_langkah or batas_langkah()), batas_langkah()), min(int(periksa), 100)


def tanda():
    return {"mode": "uji", "sisi_uji": ukuran(), "batas_langkah_uji": batas_langkah()} if aktif() else {}
