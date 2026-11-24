#!/usr/bin/env python3
"""
program/buat_gambar.py
======================
Membuat lima gambar badan laporan dan satu gambar lampiran ke folder gambar/ dari hasil tersimpan, tanpa menjalankan simulasi.
Langkah 06 dan 07 dijalankan pada salinan sementara, sehingga folder hasil/ tidak berubah.

  python program/buat_gambar.py                # dari hasil/gpu
  python program/buat_gambar.py --backend cpu  # dari hasil/cpu
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

from program.inti import inti_02_rentang_segmentasi as L02
from program.inti import inti_04_volume_elementer as L04
from program.inti import inti_06_evaluasi_kelayakan_resolusi as L06
from program.inti import inti_07_rekapitulasi_visualisasi as L07

BATUAN = ["Bentheimer", "BSG"]
HASIL = os.path.join(ROOT, "hasil", "gpu")
GAMBAR = os.path.join(ROOT, "gambar")


def baca(*bagian):
    with open(os.path.join(HASIL, *bagian), encoding="utf-8") as f:
        return json.load(f)


def main(hasil=HASIL, gambar=GAMBAR):
    global HASIL, GAMBAR
    HASIL, GAMBAR = hasil, gambar
    import matplotlib.pyplot as plt
    from program import gaya_grafik
    plt.rcdefaults()
    gaya_grafik.terapkan()   # langkah sebelumnya di proses yang sama dapat mengubah rcParams (mis. dpi)
    os.makedirs(GAMBAR, exist_ok=True)
    shutil.copy2(os.path.join(HASIL, "verifikasi", "00_verifikasi_pemecah.png"),
                 os.path.join(GAMBAR, "gambar_1_verifikasi.png"))
    L02.plot_rentang({b: baca(b, "basis_acuan.json") for b in BATUAN},
                     os.path.join(GAMBAR, "gambar_2_rentang_segmentasi.png"))
    L04.plot_rev({b: baca(b, "volume_elementer.json") for b in BATUAN},
                 os.path.join(GAMBAR, "gambar_3_volume_elementer.png"))

    with tempfile.TemporaryDirectory() as tmp:
        kerja = os.path.join(tmp, "gpu")
        os.makedirs(os.path.join(kerja, "grafik"))
        for nama in BATUAN + ["verifikasi"]:
            shutil.copytree(os.path.join(HASIL, nama), os.path.join(kerja, nama))
        with contextlib.redirect_stdout(io.StringIO()):
            L06.jalankan(rocks=BATUAN, out_dir=kerja)
            L07.jalankan(rocks=BATUAN, out_dir=kerja)
        for asal, tujuan in (("07b_riwayat_konvergensi_lbm.png", "gambar_A1_riwayat_lengkap.png"),
                             ("06b_simpangan_permeabilitas_dan_kritis.png", "gambar_4_tingkat_kritis.png"),
                             ("07a_komparasi_tiga_sifat.png", "gambar_5_tiga_sifat.png")):
            shutil.copy2(os.path.join(kerja, "grafik", asal), os.path.join(GAMBAR, tujuan))
    for nama in sorted(os.listdir(GAMBAR)):
        print(os.path.join("gambar", nama))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gambar laporan dari hasil tersimpan")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args()
    main(os.path.join(ROOT, "hasil", a.backend))
