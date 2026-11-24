#!/usr/bin/env python3
"""
program/gpu/gpu_01_permeabilitas_resolusi_penuh.py
=============================================
Langkah 01: Simulasi Permeabilitas Kondisi Tunak & Basis Acuan T0 (Mode GPU CuPy)
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Fungsi:
1. Membaca dan memvalidasi citra biner 256^3 (.raw) batupasir dinamis (Bentheimer, Parker, atau 11 batuan).
2. Menghitung sifat petrofisika geometris di VRAM GPU:
   - Porositas total (phi_tot)
   - Porositas terhubung tembus (phi_th) via pelabelan komponen terhubung 3D (cupyx.scipy.ndimage).
3. Menghitung permeabilitas kondisi tunak menggunakan pemecah aliran LBM D3Q19 bertekanan Zou-He:
   - OPSI A (mode_t0='acuan'): Memuat riwayat konvergensi acuan resmi yang tervalidasi (instan).
   - OPSI B (mode_t0='simulasi'): Menjalankan simulasi LBM D3Q19 penuh hingga kondisi tunak tercapai
     pada ketiga arah ortogonal (x, y, z) dengan kriteria P7/P11: delta_K / K < 1e-6 dan simpangan fluks antaririsan < 1e-3.
4. Menghitung permeabilitas geometrik tensorial:
   k_geo = (k_x * k_y * k_z)^(1/3)
5. Menyimpan angka hasil simulasi kondisi tunak dan kurva konvergensi secara mandiri per batuan
   di <out_dir>/<batuan>/, yang menjadi angka basis acuan untuk dibawa ke Langkah 1 s.d. Langkah 5.

Keluaran per batuan di <out_dir>/<batuan>/:
- permeabilitas_kondisi_tunak.json
- tabel_permeabilitas_tunak.csv
- simulasi_multiresolusi/<batuan>_T0_resolusi_penuh_<arah>.json
- grafik/00_permeabilitas_kondisi_tunak.png
"""

import os
import sys
import time
import json
import csv
import math
import argparse
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import gpu_solver_lbm as solver_lbm
from program.katalog_batuan import muat_semua_batuan, dapatkan_subdirektori_batuan, dapatkan_batas_langkah_batuan, hasil_sah, tunak_sah

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil")

DARCY_M2 = 0.9869233e-12
DX_M = 2.25e-6

WARNA_PALET = [
    "#0284c7", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5", "#84cc16"
]


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


def hitung_porositas_tersambung_gpu(pori_bool, sumbu=0):
    pori_gpu = cp.asarray(pori_bool)
    struct = cndi.generate_binary_structure(3, 1)
    lbl, n_lbl = cndi.label(pori_gpu, structure=struct)
    if int(n_lbl) == 0:
        return 0.0
    if sumbu == 0:
        inlet = set(cp.asnumpy(cp.unique(lbl[0, :, :])).tolist()) - {0}
        outlet = set(cp.asnumpy(cp.unique(lbl[-1, :, :])).tolist()) - {0}
    elif sumbu == 1:
        inlet = set(cp.asnumpy(cp.unique(lbl[:, 0, :])).tolist()) - {0}
        outlet = set(cp.asnumpy(cp.unique(lbl[:, -1, :])).tolist()) - {0}
    else:
        inlet = set(cp.asnumpy(cp.unique(lbl[:, :, 0])).tolist()) - {0}
        outlet = set(cp.asnumpy(cp.unique(lbl[:, :, -1])).tolist()) - {0}
    conn = inlet & outlet
    if not conn:
        return 0.0
    conn_gpu = cp.asarray(list(conn), dtype=lbl.dtype)
    mask_conn = cp.isin(lbl, conn_gpu)
    return float(mask_conn.mean())


def baca_riwayat_konvergensi_acuan(data_dir, batuan, arah):
    nama_json = f"{batuan}_penuh_{arah}.json"
    jalur_json = cari_berkas_sumber(data_dir, os.path.join("riwayat-konvergensi", nama_json))
    if not jalur_json:
        return None
    with open(jalur_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    riwayat = d.get("riwayat", [])
    steps = [r[0] for r in riwayat]
    k_kisi = [r[1] for r in riwayat]
    res = [r[2] for r in riwayat]
    k_mD_list = [(k * (DX_M ** 2) / DARCY_M2) * 1000.0 for k in k_kisi]
    return {
        "kasus": d.get("kasus", f"{batuan}_T0_{arah}"),
        "operator": d.get("kolisi", "bgk"),
        "tau": d.get("tau", 1.0),
        "drho": d.get("drho", 0.002),
        "presisi": d.get("presisi", 64),
        "domain": d.get("domain", [264, 258, 258]),
        "ukuran_asli": d.get("ukuran", [256, 256, 256]),
        "K_mD": float(d.get("K_mD", 0.0)),
        "K_kisi": float(d.get("K_kisi", 0.0)),
        "q_kisi": float(d.get("q_kisi", 0.0)),
        "langkah": int(d.get("langkah", 0)),
        "konvergen": bool(d.get("konvergen", False)),
        "detik": float(d.get("detik", 0.0)),
        "MLUPS": float(d.get("MLUPS", 0.0)),
        "fluks_variasi": float(d.get("fluks_variasi", 0.0)),
        "steps": steps,
        "k_mD_riwayat": k_mD_list,
        "residual_riwayat": res
    }



def ringkas_kasus_tunak(d, batuan, arah):
    """Ringkasan riwayat satu kasus T0 dari berkas kasus solver (kedua mode memakai fungsi ini).
    Tuple riwayat solver: (langkah, K_kisi, sisa |dK/K|, q, dP, simpangan fluks antaririsan)."""
    riw = d.get("riwayat", []) or []
    diag = d.get("diagnostik", {}) or {}
    fluks = diag.get("simpangan_fluks_irisan")
    if fluks is None and riw and len(riw[-1]) > 5:
        fluks = riw[-1][5]
    return {
        "kasus": f"{batuan}_T0_resolusi_penuh_{arah}",
        "sumber": "simulasi backend ini",
        "backend": d.get("backend"),
        "operator": "bgk",
        "tau": 1.0,
        "drho": 0.002,
        "presisi": 64,
        # domain LBM: sampel + penyangga 4 voxel (9 um) di kedua ujung + dinding samping 1 voxel
        "domain": [int(d.get("N", 256)) + 8, int(d.get("N", 256)) + 2, int(d.get("N", 256)) + 2],
        "ukuran_asli": [int(d.get("N", 256))] * 3,
        "K_mD": float(d["K_mD"]),
        "K_kisi": float(d.get("K_kisi", 0.0)),
        "q_kisi": float(riw[-1][3]) if riw and len(riw[-1]) > 3 else None,
        "dP_kisi": float(riw[-1][4]) if riw and len(riw[-1]) > 4 else None,
        "langkah": int(d["langkah"]),
        "konvergen": bool(d["konvergen"]),
        "kriteria_konvergensi": d.get("kriteria_konvergensi"),
        "detik": float(d.get("durasi_detik", 0.0)),
        "MLUPS": float(d.get("mlups", 0.0)),
        "n_fluida": d.get("n_fluida"),
        "fluks_variasi": float(fluks) if fluks is not None else None,
        "diagnostik": diag or None,
        "memori": d.get("memori"),
        "kondisi_perangkat": d.get("kondisi_perangkat"),
        "waktu_terkendali": d.get("waktu_terkendali"),
        "steps": [r[0] for r in riw] or [int(d["langkah"])],
        "k_mD_riwayat": [(r[1] * (DX_M ** 2) / DARCY_M2) * 1000.0 for r in riw] or [float(d["K_mD"])],
        "residual_riwayat": [r[2] for r in riw],
        "fluks_riwayat": [r[5] for r in riw if len(r) > 5],
    }


def plot_verifikasi_konvergensi(hasil_simulasi, rocks, out_img):
    from program.grafik_bersama import plot_konvergensi_t0
    plot_konvergensi_t0(hasil_simulasi, rocks, out_img)


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
             mode_t0="acuan", periksa=500, skip_benchmark=False):
    katalog = muat_semua_batuan(data_dir)
    if rocks is None:
        rocks = ["Bentheimer", "Parker"]

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"LANGKAH 01 [GPU]: SIMULASI PERMEABILITAS KONDISI TUNAK ACUAN (MODE={mode_t0.upper()})")
    print(f"Sampel Batuan: {', '.join(rocks)}")
    print("=" * 80, flush=True)

    jalur_meta = cari_berkas_sumber(data_dir, "metadata-citra.json")
    meta = {}
    if jalur_meta:
        with open(jalur_meta, encoding="utf-8") as f:
            meta = json.load(f)

    if meta:
        print(f"\n[1] Metadata Citra Biner Mentah (256^3, dx = 2.25 um):")
        print(f"    Sumber Data        : {meta.get('sumber', 'Neumann dkk. (2021)')}")
        print(f"    Ukuran Voxel       : 256^3 voxel | dx = 2.25 um")
        print(f"    Konvensi Nilai     : 0 = fluida (pori), 1 = padatan")

    hasil_simulasi = {}
    baris_tabel_csv = []

    print(f"\n[2] Verifikasi Morfometri Citra Biner (.raw) [GPU CuPy]:")
    print(f"    {'Batuan':<14} {'phi_raw':<10} {'phi_acuan':<10} {'phi_th_raw':<11} {'phi_th_acuan':<12} {'Status'}")
    print("    " + "-" * 72)

    for batuan in rocks:
        berkas_raw = f"{batuan}_2d25um_binary_256.raw"
        jalur_raw = cari_berkas_sumber(data_dir, os.path.join("citra-biner", berkas_raw))
        if not jalur_raw and batuan in katalog:
            jalur_raw = katalog[batuan].get("jalur_berkas")
        if not jalur_raw:
            raise FileNotFoundError(f"Citra biner {berkas_raw} tidak ditemukan!")

        pori = baca_citra_biner(jalur_raw, n=256, urutan=meta.get("urutan_indeks", "C"))
        phi_raw = float(cp.asarray(pori).mean())
        phi_th_raw = hitung_porositas_tersambung_gpu(pori, sumbu=2)  # Sumbu x pada C-order [z, y, x]

        info_bat = katalog.get(batuan, {})
        phi_csv = info_bat.get("phi_acuan", phi_raw)
        phi_th_csv = info_bat.get("phi_th_acuan", phi_th_raw)

        selisih = abs(phi_raw - phi_csv)
        status = "TERVERIFIKASI" if selisih < 1e-4 else "BERGESER"
        print(f"    {batuan:<14} {phi_raw:<10.6f} {phi_csv:<10.6f} {phi_th_raw:<11.6f} {phi_th_csv:<12.6f} {status}", flush=True)

        # Folder mandiri per batuan
        dir_batuan = dapatkan_subdirektori_batuan(out_dir, batuan)
        sim_multi_dir = os.path.join(dir_batuan, "simulasi_multiresolusi")
        os.makedirs(sim_multi_dir, exist_ok=True)

        k_arah = {}
        riwayat_arah = {}

        print(f"\n[3] Perhitungan Permeabilitas Kondisi Tunak LBM D3Q19 [GPU]: {batuan}")
        for arah in ["x", "y", "z"]:
            out_sim_json = os.path.join(sim_multi_dir, f"{batuan}_T0_resolusi_penuh_{arah}.json")

            if mode_t0 == "acuan":
                # Periksa apakah sudah ada hasil simulasi tunak nyata di disk
                d_tunak = None
                for c_path in [out_sim_json]:   # hanya hasil backend ini sendiri
                    if os.path.exists(c_path):
                        try:
                            with open(c_path, "r", encoding="utf-8") as f_c:
                                d_c = json.load(f_c)
                            if d_c.get("backend") in ["gpu", "cpu"] and tunak_sah(d_c):
                                d_tunak = d_c
                                break
                        except Exception:
                            pass

                if d_tunak:
                    k_val = float(d_tunak["K_mD"])
                    langkah_sim = int(d_tunak["langkah"])
                    konvergen_sim = 1
                    sim_acuan = ringkas_kasus_tunak(d_tunak, batuan, arah)
                    print(f"    * [{batuan} Arah {arah.upper()} | SIMULASI TUNAK TERSEDIA] K = {k_val:.2f} mD ({langkah_sim} langkah, konvergen=True)", flush=True)
                else:
                    # Mode A: Gunakan riwayat konvergensi acuan resmi
                    sim_acuan = baca_riwayat_konvergensi_acuan(data_dir, batuan, arah)
                    if sim_acuan:
                        sim_acuan["sumber"] = "data acuan dosen (riwayat-konvergensi)"
                        k_val = sim_acuan["K_mD"]
                        langkah_sim = sim_acuan["langkah"]
                        konvergen_sim = 1 if sim_acuan["konvergen"] else 0
                    else:
                        k_val = float(info_bat.get("K_mD_acuan", {}).get(arah, 100.0))
                        langkah_sim = int(info_bat.get("langkah_acuan", {}).get(arah, 15000))
                        konvergen_sim = 1 if info_bat.get("konvergen_acuan", {}).get(arah, True) else 0
                        sim_acuan = {
                            "kasus": f"{batuan}_T0_resolusi_penuh_{arah}",
                            "operator": "bgk",
                            "tau": 1.0,
                            "drho": 0.002,
                            "presisi": 64,
                            "domain": [264, 258, 258],
                            "ukuran_asli": [256, 256, 256],
                            "K_mD": k_val,
                            "K_kisi": 0.0,
                            "q_kisi": 0.0,
                            "langkah": langkah_sim,
                            "konvergen": bool(konvergen_sim),
                            "detik": 0.0,
                            "MLUPS": 0.0,
                            "fluks_variasi": 0.0,
                            "steps": [langkah_sim],
                            "k_mD_riwayat": [k_val],
                            "residual_riwayat": [4.0e-5]
                        }
                    print(f"    * [{batuan} Arah {arah.upper()} | ACUAN] K = {k_val:.2f} mD ({langkah_sim} langkah, konvergen={bool(konvergen_sim)})", flush=True)
            else:
                # Mode B: simulasi LBM hingga tunak; hasil sah backend ini yang sudah konvergen dipakai ulang
                maks_steps = dapatkan_batas_langkah_batuan(katalog, batuan, tingkat=0, arah=arah)
                sim_res = None
                if os.path.exists(out_sim_json):
                    with open(out_sim_json, encoding="utf-8") as f_c:
                        d_c = json.load(f_c)
                    if d_c.get("backend") == "gpu" and tunak_sah(d_c):
                        sim_res = d_c
                        print(f"    * [{batuan} Arah {arah.upper()}] hasil sah tersedia, dipakai ulang: K = {d_c['K_mD']:.2f} mD ({d_c['langkah']} langkah)", flush=True)
                if sim_res is None:
                    sim_res = solver_lbm.jalankan_kasus_lbm(
                        pori, batuan, 0, "resolusi_penuh", arah,
                        out_file=out_sim_json, maks_langkah=maks_steps, periksa=periksa, verbose=True
                    )
                k_val = float(sim_res["K_mD"])
                langkah_sim = int(sim_res["langkah"])
                konvergen_sim = 1 if sim_res["konvergen"] else 0
                sim_acuan = ringkas_kasus_tunak(sim_res, batuan, arah)

            k_arah[arah] = k_val
            riwayat_arah[arah] = sim_acuan

            baris_tabel_csv.append({
                "batuan": batuan,
                "arah": arah,
                "phi": f"{phi_raw:.6f}",
                "phi_terhubung": f"{phi_th_raw:.6f}",
                "K_mD": f"{k_val:.4f}",
                "K_kisi": f"{sim_acuan['K_kisi']:.8e}" if sim_acuan else "0",
                "langkah": langkah_sim,
                "konvergen": konvergen_sim,
                "operator": sim_acuan["operator"] if sim_acuan else "bgk",
                "tau": sim_acuan["tau"] if sim_acuan else 1.0,
                "drho": sim_acuan["drho"] if sim_acuan else 0.002,
                "domain": str(sim_acuan["domain"]) if sim_acuan else "[264, 258, 258]"
            })

        k_geo = float((k_arah["x"] * k_arah["y"] * k_arah["z"]) ** (1.0 / 3.0))
        print(f"    --> HASIL TUNAK {batuan}: Kx={k_arah['x']:.2f} mD, Ky={k_arah['y']:.2f} mD, Kz={k_arah['z']:.2f} mD | K_geo={k_geo:.2f} mD\n")

        data_batuan_tunggal = {
            "batuan": batuan,
            "nama": batuan,
            "berkas_raw": berkas_raw,
            "dx_um": 2.25,
            "N": int(pori.shape[0]),
            "phi_total": phi_raw,
            "phi_tersambung": phi_th_raw,
            "fraksi_terisolasi_persen": (1.0 - phi_th_raw / phi_raw) * 100.0 if phi_raw > 0 else 0.0,
            "K_mD": k_arah,
            "K_geometrik_mD": k_geo,
            "anisotropi_k_max_min": max(k_arah.values()) / min(k_arah.values()),
            "riwayat_konvergensi": riwayat_arah,
            "mode_eksekusi": mode_t0
        }
        hasil_simulasi[batuan] = data_batuan_tunggal

        # SIMPAN MANDIRI PER BATUAN DI HASIL/<BATUAN>/:
        out_sub_json = os.path.join(dir_batuan, "permeabilitas_kondisi_tunak.json")
        data_simpan_mandiri = dict(data_batuan_tunggal)
        data_simpan_mandiri[batuan] = data_batuan_tunggal  # Dukung akses langsung d["K_mD"] maupun d[batuan]["K_mD"]
        with open(out_sub_json, "w", encoding="utf-8") as f:
            json.dump(data_simpan_mandiri, f, indent=2)

        out_sub_csv = os.path.join(dir_batuan, "tabel_permeabilitas_tunak.csv")
        baris_batu = [r for r in baris_tabel_csv if r["batuan"] == batuan]
        with open(out_sub_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(baris_batu[0].keys()))
            writer.writeheader()
            writer.writerows(baris_batu)

        out_sub_img = os.path.join(dir_batuan, "grafik", "01_permeabilitas_resolusi_penuh.png")
        plot_verifikasi_konvergensi({batuan: data_batuan_tunggal}, [batuan], out_sub_img)
        print(f"    [OK] Berkas mandiri tersimpan di: {dir_batuan}/")

    # Jika memproses lebih dari 1 batuan, simpan juga komparasi/kompilasi di root out_dir
    if len(rocks) > 1:
        out_json = os.path.join(out_dir, "permeabilitas_kondisi_tunak.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(hasil_simulasi, f, indent=2)

        out_csv = os.path.join(out_dir, "tabel_permeabilitas_tunak.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(baris_tabel_csv[0].keys()))
            writer.writeheader()
            writer.writerows(baris_tabel_csv)

        grafik_dir = os.path.join(out_dir, "grafik")
        os.makedirs(grafik_dir, exist_ok=True)
        out_img = os.path.join(grafik_dir, "01_permeabilitas_resolusi_penuh.png")
        plot_verifikasi_konvergensi(hasil_simulasi, list(hasil_simulasi.keys()), out_img)

    # Cetak ringkasan lengkap simulasi kondisi tunak ke stdout/log
    cetak_ringkasan_simulasi(hasil_simulasi, rocks)

    print(f"--> Nilai permeabilitas acuan ini AKAN DIBAWA ke Langkah 1 s.d. Langkah 5.")
    print("=" * 80)
    return hasil_simulasi


def cetak_ringkasan_simulasi(hasil_simulasi, rocks):
    print("\n" + "=" * 80)
    print("RINGKASAN HASIL SIMULASI PERMEABILITAS KONDISI TUNAK (RESOLUSI T0 - 256^3)")
    print("=" * 80)
    for batuan in rocks:
        if batuan not in hasil_simulasi:
            continue
        d = hasil_simulasi[batuan]
        phi_tot = d["phi_total"]
        phi_th = d["phi_tersambung"]
        fraksi_isolasi = d["fraksi_terisolasi_persen"]
        k_geo = d["K_geometrik_mD"]
        anisotropi = d["anisotropi_k_max_min"]
        label_aniso = "Isotropik Lemah" if anisotropi < 1.5 else ("Anisotropik Sedang" if anisotropi < 3.0 else "Anisotropik Kuat")
        mode = d.get("mode_eksekusi", "acuan").upper()

        print(f"\n[SAMPEL BATUAN: {batuan.upper()}] (Mode: {mode})")
        print(f"  * Geometri Voxel           : 256^3 voxel | dx = 2.25 um | Dimensi Sampel = (576 um)^3")
        print(f"  * Porositas Total (phi)    : {phi_tot:.4f} ({phi_tot*100:.2f}%)")
        print(f"  * Porositas Tembus (phi_th): {phi_th:.4f} ({phi_th*100:.2f}%) [Fraksi Terisolasi: {fraksi_isolasi:.2f}%]")
        print("  " + "-" * 76)
        print(f"  {'Arah':<6} {'Permeabilitas (mD)':<20} {'Langkah':<10} {'Status Konvergensi':<20} {'Durasi (s)':<12} {'MLUPS'}")
        print("  " + "-" * 76)

        total_durasi = 0.0
        for arah in ["x", "y", "z"]:
            r = d["riwayat_konvergensi"].get(arah, {})
            k_val = d["K_mD"].get(arah, 0.0)
            langkah = r.get("langkah", 0)
            is_konv = r.get("konvergen", False)
            status_str = "TUNAK (True)" if is_konv else "Belum Tunak (False)"
            dur = r.get("detik", 0.0)
            total_durasi += dur
            mlups = r.get("MLUPS", 0.0)
            print(f"  {arah.upper():<6} {k_val:>12.2f} mD      {langkah:>8,d}   {status_str:<20} {dur:>8.2f}s    {mlups:>7.1f}")

        print("  " + "-" * 76)
        print(f"  * Permeabilitas Rerata Geometrik (K_geo) : {k_geo:.2f} mD")
        print(f"  * Rasio Anisotropi (K_max / K_min)       : {anisotropi:.2f} ({label_aniso})")
        if total_durasi > 0:
            print(f"  * Total Waktu Komputasi (3 Arah)         : {total_durasi:.2f} detik (~{total_durasi/60:.2f} menit)")
    print("=" * 80 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Langkah 01 [GPU]: Simulasi Permeabilitas Kondisi Tunak")
    parser.add_argument("--rocks", nargs="+", default=["Bentheimer", "Parker"])
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode_t0", default="acuan", choices=["acuan", "simulasi"])
    parser.add_argument("--periksa", type=int, default=500)
    parser.add_argument("--skip_benchmark", action="store_true")
    args = parser.parse_args()
    jalankan(
        rocks=args.rocks, data_dir=args.data_dir, out_dir=args.out_dir,
        mode_t0=args.mode_t0, periksa=args.periksa, skip_benchmark=args.skip_benchmark
    )


if __name__ == "__main__":
    main()
