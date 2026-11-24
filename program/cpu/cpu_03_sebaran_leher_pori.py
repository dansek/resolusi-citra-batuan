#!/usr/bin/env python3
"""
program/cpu/cpu_03_sebaran_leher_pori.py
====================================
Langkah 03: Karakteristik Distribusi Leher Pori (d_t, d_c, Kriteria Nyquist) [Mode CPU NumPy/SciPy]
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Tujuan:
1. Menghitung Euclidean Distance Transform (EDT 3D) di CPU (scipy.ndimage.distance_transform_edt)
   untuk memperoleh jari-jari pori maksimum lokal.
2. Mengekstrak struktur tulang punggung ruang pori (Skeleton 3D) via skimage.morphology.skeletonize_3d.
3. Menghitung sebaran diameter leher pori d_c (median, rerata, d10, d90) pada simpul-simpul skeleton.
4. Menjalankan granulometri berbasis EDT 3D di CPU untuk menentukan diameter pori pengendali permeabilitas:
   - Intrusi bola uji r = 1..r_max
   - Evaluasi ambang 50% ruang pori tersambung: r_50 -> d_t = 2 * r_50
5. Menguji kriteria keterukuran resolusi voxel (dx = 2.25 um):
   - Batas Nyquist: dx_maks = d_t / 2
   - Kaidah 4-Voxel: dx_kaidah = d_t / 4

Keluaran per batuan di <out_dir>/<batuan>/:
- karakteristik_leher_pori.json
- tabel_karakteristik_leher_pori.csv
- grafik/02_distribusi_leher_pori.png
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.katalog_batuan import muat_semua_batuan, dapatkan_subdirektori_batuan
from program.pemuat_modul import muat as _muat
_lp = _muat("program/inti/inti_03_leher_penentu.py")
try:
    from program.utilitas_hardware import terapkan_afinitas_core_fisik
    terapkan_afinitas_core_fisik()
except Exception:
    pass

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil")

VOXEL_UM = 2.25

WARNA_PALET = [
    "#0284c7", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5", "#84cc16"
]
MARKER_PALET = ["o", "s", "^", "v", "D", "p", "*", "<", ">", "h", "H"]


def cari_berkas_sumber(data_dir, subpath):
    kandidat = [
        os.path.join(data_dir, "sumber", subpath),
        os.path.join(data_dir, subpath),
        os.path.join(ROOT, "data", "sumber", subpath),
        os.path.join(ROOT, "data", subpath)
    ]
    for k in kandidat:
        if os.path.exists(k):
            return k
    return None


def baca_citra_biner(jalur_raw, n=256, urutan="C"):
    from program.citra import baca_citra_biner as _baca
    return _baca(jalur_raw, n=n, urutan=urutan)


def granulometri_p2_cpu(pori_bool, edt=None, voxel_um=2.25, jari_maks=25.0, langkah=1.0):
    if edt is None:
        edt = ndi.distance_transform_edt(pori_bool)

    total_pori = int(np.count_nonzero(pori_bool))
    if total_pori == 0:
        return [0.0], [0.0], 0.0, 0.0

    jari_list = np.arange(1.0, jari_maks + langkah * 0.5, langkah)
    bagian = []
    jari_um = []

    for r_um in jari_list:
        r_vox = r_um / voxel_um
        mask_r = (edt >= r_vox)
        edt_inv = ndi.distance_transform_edt(~mask_r)
        rekon = (edt_inv <= r_vox) & pori_bool
        vol_rekon = int(np.count_nonzero(rekon))
        fraksi = vol_rekon / total_pori
        bagian.append(fraksi)
        jari_um.append(float(r_um))

    r50_idx = np.where(np.array(bagian) <= 0.50)[0]
    if len(r50_idx) > 0:
        idx = r50_idx[0]
        if idx > 0:
            r1, r2 = jari_um[idx - 1], jari_um[idx]
            f1, f2 = bagian[idx - 1], bagian[idx]
            if f1 != f2:
                r50_um = r1 + (0.50 - f1) * (r2 - r1) / (f2 - f1)
            else:
                r50_um = (r1 + r2) / 2.0
        else:
            r50_um = jari_um[0]
    else:
        r50_um = float(jari_um[-1])

    dt_um = 2.0 * r50_um
    return jari_um, bagian, r50_um, dt_um


def plot_leher_pori(hasil_leher_pori, data_histogram, out_img):
    from program.grafik_bersama import plot_leher_pori as _plot
    _plot(hasil_leher_pori, data_histogram, out_img)


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    katalog = muat_semua_batuan(data_dir)
    if rocks is None:
        rocks = ["Bentheimer", "Parker"]

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("LANGKAH 03 [CPU]: KARAKTERISTIK DISTRIBUSI LEHER PORI (d_t, d_c, NYQUIST)")
    print(f"Sampel Batuan: {', '.join(rocks)}")
    print("=" * 80, flush=True)

    hasil_leher_pori = {}
    data_histogram = {}

    for kunci in rocks:
        info_bat = katalog.get(kunci, {})
        nama = info_bat.get("nama", f"{kunci} Sandstone")
        tipe = info_bat.get("tipe", "Batupasir")
        berkas = info_bat.get("berkas", f"{kunci}_2d25um_binary_256.raw")
        jalur = cari_berkas_sumber(data_dir, os.path.join("citra-biner", berkas))
        if not jalur:
            jalur = info_bat.get("jalur_berkas")

        print(f"\n>>> Memproses Batuan: {nama} ({tipe}) [CPU NumPy]", flush=True)
        if not jalur or not os.path.exists(jalur):
            raise FileNotFoundError(f"Berkas {berkas} tidak ditemukan!")

        pori = baca_citra_biner(jalur, n=256)
        phi_total = float(pori.mean())

        t0 = time.time()
        print("    [1/3] Menghitung Euclidean Distance Transform (EDT) 3D di CPU...", flush=True)
        edt = ndi.distance_transform_edt(pori)
        r_maks_voxel = float(edt.max())
        r_maks_um = r_maks_voxel * VOXEL_UM
        print(f"          Jari-jari pori maksimum = {r_maks_voxel:.2f} voxel ({r_maks_um:.2f} um) ({time.time()-t0:.2f}s)")

        t0 = time.time()
        print("    [2/3] Mengekstrak tulang punggung ruang pori (Skeleton 3D)...", flush=True)
        skel = skeletonize(pori)
        cacah_simpul_skel = int(np.count_nonzero(skel))
        r_skel = edt[skel]
        d_skel_um = 2.0 * r_skel * VOXEL_UM

        d10_skel = float(np.percentile(d_skel_um, 10))
        d50_skel = float(np.median(d_skel_um))
        d90_skel = float(np.percentile(d_skel_um, 90))
        d_mean_skel = float(np.mean(d_skel_um))
        d_min_skel = float(np.min(d_skel_um))
        d_max_skel = float(np.max(d_skel_um))
        print(f"          Simpul skeleton: {cacah_simpul_skel:,} voxel | d10={d10_skel:.2f} um, d_c (median)={d50_skel:.2f} um, d90={d90_skel:.2f} um ({time.time()-t0:.2f}s)")

        t0 = time.time()
        print("    [3/3] Menjalankan Granulometri EDT 3D di CPU (tiruan intrusi raksa)...", flush=True)
        jari_maks_ukur = float(max(15.0, min(r_maks_um, 35.0)))
        jari_um, bagian_kurva, r50_granulo, dt_granulo = granulometri_p2_cpu(
            pori, edt=edt, voxel_um=VOXEL_UM, jari_maks=jari_maks_ukur, langkah=1.0
        )
        # Leher PENENTU per arah (takrif P16/P11), dipakai sebagai d_t utama Bab 8
        lp = _lp.leher_penentu(pori, VOXEL_UM, edt=edt)
        dt_penentu = lp["d_t_um"]
        print(f"          Leher penentu P16 : d_c x/y/z = {lp['d_c_um']['x']:.2f} / {lp['d_c_um']['y']:.2f} / "
              f"{lp['d_c_um']['z']:.2f} um -> d_t = {dt_penentu:.2f} um (arah pembatas {lp['arah_pembatas']})", flush=True)
        dx_nyquist = dt_penentu / 2.0
        dx_4voxel = dt_penentu / 4.0
        print(f"          Hasil Granulometri: r_50 = {r50_granulo:.2f} um -> d_t = {dt_granulo:.2f} um ({time.time()-t0:.2f}s)")
        print(f"          Kriteria Resolusi : Batas Nyquist (dx_maks = d_t/2) = {dx_nyquist:.2f} um | Kaidah 4-Voxel (d_t/4) = {dx_4voxel:.2f} um")

        data_batuan = {
            "batuan": kunci,
            "nama": nama,
            "tipe": tipe,
            "phi_total": phi_total,
            "r_maks_um": r_maks_um,
            "dt_um": dt_penentu,
            "takrif_dt": lp["takrif"],
            "leher_penentu": lp,
            "dt_granulometri_um": dt_granulo,
            "r50_granulometri_um": r50_granulo,
            "dc_skeleton_median_um": d50_skel,
            "dx_nyquist_um": dx_nyquist,
            "dx_kaidah_4voxel_um": dx_4voxel,
            "d_skeleton": {
                "d_median_um": d50_skel,
                "d_mean_um": d_mean_skel,
                "d10_um": d10_skel,
                "d90_um": d90_skel,
                "d_min_um": d_min_skel,
                "d_max_um": d_max_skel,
                "cacah_simpul": cacah_simpul_skel
            },
            "kurva_granulometri": {
                "jari_um": jari_um,
                "fraksi_volume": bagian_kurva
            }
        }
        hasil_leher_pori[kunci] = data_batuan
        data_histogram[kunci] = d_skel_um

        # SIMPAN MANDIRI DI HASIL/<BATUAN>/:
        dir_batuan = dapatkan_subdirektori_batuan(out_dir, kunci)
        out_sub_json = os.path.join(dir_batuan, "karakteristik_leher_pori.json")
        data_mandiri = dict(data_batuan)
        data_mandiri[kunci] = data_batuan
        with open(out_sub_json, "w", encoding="utf-8") as f:
            json.dump(data_mandiri, f, indent=2)

        out_sub_csv = os.path.join(dir_batuan, "tabel_karakteristik_leher_pori.csv")
        baris_csv_batu = [{
            "batuan": kunci,
            "nama": nama,
            "phi_total": f"{phi_total:.4f}",
            "r_maks_um": f"{r_maks_um:.2f}",
            "r50_granulometri_um": f"{r50_granulo:.2f}",
            "dt_granulometri_um": f"{dt_granulo:.2f}",
            "dc_skeleton_median_um": f"{d50_skel:.2f}",
            "d10_skeleton_um": f"{d10_skel:.2f}",
            "d90_skeleton_um": f"{d90_skel:.2f}",
            "dx_nyquist_um": f"{dx_nyquist:.2f}",
            "dx_kaidah_4voxel_um": f"{dx_4voxel:.2f}",
            "dt_penentu_um": f"{dt_penentu:.2f}",
            "dc_penentu_x_um": f"{lp['d_c_um']['x']:.2f}",
            "dc_penentu_y_um": f"{lp['d_c_um']['y']:.2f}",
            "dc_penentu_z_um": f"{lp['d_c_um']['z']:.2f}"
        }]
        with open(out_sub_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(baris_csv_batu[0].keys()))
            writer.writeheader()
            writer.writerows(baris_csv_batu)

        out_sub_img = os.path.join(dir_batuan, "grafik", "03_distribusi_leher_pori.png")
        plot_leher_pori({kunci: data_batuan}, {kunci: d_skel_um}, out_sub_img)
        print(f"    [OK] Berkas mandiri tersimpan di: {dir_batuan}/")

    # Simpan komparasi di root out_dir jika memproses > 1 batuan
    if len(rocks) > 1:
        out_json = os.path.join(out_dir, "karakteristik_leher_pori.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(hasil_leher_pori, f, indent=2)

        baris_csv_gabungan = []
        for k_rock, d in hasil_leher_pori.items():
            baris_csv_gabungan.append({
                "batuan": k_rock,
                "nama": d["nama"],
                "phi_total": f"{d['phi_total']:.4f}",
                "r_maks_um": f"{d['r_maks_um']:.2f}",
                "r50_granulometri_um": f"{d['r50_granulometri_um']:.2f}",
                "dt_granulometri_um": f"{d['dt_granulometri_um']:.2f}",
                "dc_skeleton_median_um": f"{d['dc_skeleton_median_um']:.2f}",
                "d10_skeleton_um": f"{d['d_skeleton']['d10_um']:.2f}",
                "d90_skeleton_um": f"{d['d_skeleton']['d90_um']:.2f}",
                "dx_nyquist_um": f"{d['dx_nyquist_um']:.2f}",
                "dx_kaidah_4voxel_um": f"{d['dx_kaidah_4voxel_um']:.2f}",
                "dt_penentu_um": f"{d['dt_um']:.2f}",
                "dc_penentu_x_um": f"{d['leher_penentu']['d_c_um']['x']:.2f}",
                "dc_penentu_y_um": f"{d['leher_penentu']['d_c_um']['y']:.2f}",
                "dc_penentu_z_um": f"{d['leher_penentu']['d_c_um']['z']:.2f}"
            })

        out_csv = os.path.join(out_dir, "tabel_karakteristik_leher_pori.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(baris_csv_gabungan[0].keys()))
            writer.writeheader()
            writer.writerows(baris_csv_gabungan)

        grafik_dir = os.path.join(out_dir, "grafik")
        os.makedirs(grafik_dir, exist_ok=True)
        out_plot = os.path.join(grafik_dir, "03_distribusi_leher_pori.png")
        plot_leher_pori(hasil_leher_pori, data_histogram, out_plot)

    print("\n" + "=" * 80)
    print("SELESAI LANGKAH 03: Seluruh karakteristik leher pori tersimpan.")
    print("=" * 80)
    return hasil_leher_pori


def main():
    parser = argparse.ArgumentParser(description="Langkah 03 [CPU]: Karakteristik Distribusi Leher Pori")
    parser.add_argument("--rocks", nargs="+", default=["Bentheimer", "Parker"])
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    jalankan(rocks=args.rocks, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
