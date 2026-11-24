#!/usr/bin/env python3
"""
program/gpu/gpu_05_multiresolusi_lbm.py
===================================
Langkah 05: Multiresolusi (T0-T3) & Simulasi Aliran LBM D3Q19 [Mode GPU CuPy]
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Tujuan:
1. Membentuk citra multiresolusi 4 tingkat (T0: 256^3, T1: 128^3, T2: 64^3, T3: 32^3)
   menggunakan 2 operator reduksi spasial berakselerasi GPU:
   - Suara terbanyak pada blok f^3: pori bila cacah pori > f^3/2; keadaan SERI (tepat f^3/2) -> PADATAN.
   - Rerata blok diambangkan ulang: medan pecahan rerata blok diambangkan pada tingkat T_t yang
     dipilih ulang per tingkat sehingga porositas total paling dekat dengan porositas T0
     (Bab 8, jebakan 'ambang yang diwariskan apa adanya'). T_t dicatat pada tabel.
2. Menghitung morfometri tiap tingkat resolusi:
   - Porositas total (phi_tot) dan porositas tembus (phi_th) di GPU
   - Luas permukaan spesifik (Sv) di GPU
   - Rasio resolusi fisik terhadap leher pori (d_t / dx)
3. Menjalankan simulasi aliran LBM D3Q19 bertekanan Zou-He (tau=1.0, drho=0.002):
   - Resolusi Penuh T0: Sesuai mode_t0 ('acuan' memuat hasil resmi, 'simulasi' LBM hingga tunak)
   - Resolusi T1, T2, T3: Simulasi 3 arah (x, y, z) untuk kedua operator reduksi.

Keluaran per batuan di <out_dir>/<batuan>/:
- multiresolusi/<batuan>_T<tingkat>_<cara>.npy
- simulasi_multiresolusi/<batuan>_T<tingkat>_<cara>_<arah>.json
- tabel_multiresolusi.csv
- multiresolusi_dan_lbm.json
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cndi

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


def golongkan_tembus_gpu(pori_bool_gpu, sumbu=0):
    struct = cndi.generate_binary_structure(3, 1)
    lbl, n_lbl = cndi.label(pori_bool_gpu, structure=struct)
    if int(n_lbl) == 0:
        return cp.zeros_like(pori_bool_gpu, dtype=bool)

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
        return cp.zeros_like(pori_bool_gpu, dtype=bool)

    conn_gpu = cp.asarray(list(conn), dtype=lbl.dtype)
    return cp.isin(lbl, conn_gpu)


def hitung_luas_spesifik_gpu(pori_bool_gpu, dx_m):
    p_int = pori_bool_gpu.astype(cp.int8)
    diff_x = cp.abs(p_int[1:, :, :] - p_int[:-1, :, :])
    diff_y = cp.abs(p_int[:, 1:, :] - p_int[:, :-1, :])
    diff_z = cp.abs(p_int[:, :, 1:] - p_int[:, :, :-1])
    luas_voxel = float(diff_x.sum() + diff_y.sum() + diff_z.sum())
    volume_total_m3 = (pori_bool_gpu.shape[0] * dx_m) ** 3
    luas_total_m2 = luas_voxel * (dx_m ** 2)
    Sv_m_inv = luas_total_m2 / volume_total_m3
    return Sv_m_inv * 1e-6


def turunkan_suara_terbanyak_gpu(vol_gpu, faktor):
    if faktor == 1:
        return vol_gpu.copy()
    m = vol_gpu.shape[0] // faktor
    b = vol_gpu.reshape(m, faktor, m, faktor, m, faktor)
    counts = b.sum(axis=(1, 3, 5))
    return (counts > (faktor ** 3 / 2.0))


def turunkan_rerata_blok_gpu(vol, faktor, phi_target):
    """Rerata blok lalu diambangkan ulang pada T_t yang menjaga porositas T0 (nilai diskret k/f^3)."""
    if faktor == 1:
        return vol.copy(), 0.5
    m = vol.shape[0] // faktor
    avg = vol.astype(cp.float32).reshape(m, faktor, m, faktor, m, faktor).mean(axis=(1, 3, 5))
    calon = [(k + 0.5) / faktor ** 3 for k in range(faktor ** 3)]          # di antara nilai diskret
    phi_calon = [float((avg > T).mean()) for T in calon]
    j = int(np.argmin([abs(ph - phi_target) for ph in phi_calon]))
    return (avg > calon[j]), float(calon[j])


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, mode_t0="acuan", periksa=500, skip_benchmark=False):
    katalog = muat_semua_batuan(data_dir)
    if rocks is None:
        rocks = ["Bentheimer", "Parker"]

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"LANGKAH 05 [GPU]: MULTIRESOLUSI (T0-T3) & SIMULASI ALIRAN LBM D3Q19 (T0={mode_t0.upper()})")
    print(f"Sampel Batuan: {', '.join(rocks)}")
    print("=" * 80, flush=True)

    TINGKAT_RESOLUSI = [
        {"tingkat": 0, "kode": "T0", "faktor": 1, "N": 256, "dx": 2.25, "nama_tingkat": "T0 (256^3)"},
        {"tingkat": 1, "kode": "T1", "faktor": 2, "N": 128, "dx": 4.50, "nama_tingkat": "T1 (128^3)"},
        {"tingkat": 2, "kode": "T2", "faktor": 4, "N": 64,  "dx": 9.00, "nama_tingkat": "T2 (64^3)"},
        {"tingkat": 3, "kode": "T3", "faktor": 8, "N": 32,  "dx": 18.00, "nama_tingkat": "T3 (32^3)"}
    ]

    volume_cache = {}
    semua_baris_global = []

    for kunci in rocks:
        info_bat = katalog.get(kunci, {})
        dir_batuan = dapatkan_subdirektori_batuan(out_dir, kunci)
        multi_dir = os.path.join(dir_batuan, "multiresolusi")
        sim_dir = os.path.join(dir_batuan, "simulasi_multiresolusi")
        os.makedirs(multi_dir, exist_ok=True)
        os.makedirs(sim_dir, exist_ok=True)

        # Cari d_t dari Langkah 2
        dt_val = None
        for j_leher in [os.path.join(dir_batuan, "karakteristik_leher_pori.json"), os.path.join(out_dir, "karakteristik_leher_pori.json")]:
            if os.path.exists(j_leher):
                try:
                    with open(j_leher, encoding="utf-8") as f:
                        d_lh = json.load(f)
                    # d_t utama = leher penentu P16/P11 (Langkah 03)
                    if kunci in d_lh and "dt_um" in d_lh[kunci]:
                        dt_val = float(d_lh[kunci]["dt_um"])
                    elif "dt_um" in d_lh:
                        dt_val = float(d_lh["dt_um"])
                    if dt_val:
                        break
                except Exception:
                    pass

        if dt_val is None:
            raise FileNotFoundError(f"karakteristik_leher_pori.json untuk {kunci} tidak ada; jalankan Langkah 03 lebih dahulu")

        berkas_raw = info_bat.get("berkas", f"{kunci}_2d25um_binary_256.raw")
        jalur_raw = cari_berkas_sumber(data_dir, os.path.join("citra-biner", berkas_raw))
        if not jalur_raw:
            jalur_raw = info_bat.get("jalur_berkas")
        if not jalur_raw or not os.path.exists(jalur_raw):
            raise FileNotFoundError(f"Berkas {berkas_raw} tidak ditemukan!")

        pori_penuh = baca_citra_biner(jalur_raw, n=256)
        pori_penuh_gpu = cp.asarray(pori_penuh)
        volume_cache[kunci] = {}

        print(f"\n>>> [1/2] Pembentukan Citra Multiresolusi [GPU CuPy]: {kunci} (d_t = {dt_val:.2f} um)")
        print(f"    {'Tingkat':<12} {'Metode':<27} {'N^3':<6} {'dx(um)':<8} {'phi_tot':<10} {'phi_th_x':<10} {'Sv(um^-1)':<11} {'dt/dx':<8}")
        print("    " + "-" * 95)

        baris_batu = []
        for ting in TINGKAT_RESOLUSI:
            t_idx = ting["tingkat"]
            f = ting["faktor"]
            N = int(pori_penuh.shape[0]) // f   # 256/f; potongan uji ikut benar
            dx = ting["dx"]
            nisbah = dt_val / dx

            if t_idx == 0:
                daftar_cara = [("resolusi_penuh", "Resolusi Penuh (256^3)")]
            else:
                daftar_cara = [
                    ("majority", "Suara terbanyak (seri = padatan)"),
                    ("block_avg", "Rerata blok (ambang ulang, phi T0)")
                ]

            for cara, nama_met in daftar_cara:
                ambang_cara = 0.5
                ambang_blok = None
                if cara == "resolusi_penuh":
                    vol_gpu = pori_penuh_gpu
                elif cara == "majority":
                    vol_gpu = turunkan_suara_terbanyak_gpu(pori_penuh_gpu, f)
                else:
                    vol_gpu, ambang_blok = turunkan_rerata_blok_gpu(pori_penuh_gpu, f, phi_target=float(pori_penuh_gpu.mean()))

                if ambang_blok is not None:
                    ambang_cara = ambang_blok
                volume_cache[kunci][(t_idx, cara)] = vol_gpu

                phi_tot = float(vol_gpu.mean())
                tembus_x_gpu = golongkan_tembus_gpu(vol_gpu, sumbu=2)  # Sumbu x pada C-order [z, y, x]
                phi_th = float(tembus_x_gpu.mean())
                Sv = hitung_luas_spesifik_gpu(vol_gpu, dx * 1e-6)

                nama_npy = f"{kunci}_T{t_idx}_{cara}.npy"
                jalur_npy = os.path.join(multi_dir, nama_npy)
                np.save(jalur_npy, cp.asnumpy(vol_gpu))

                row_entry = {
                    "batuan": kunci,
                    "tingkat": t_idx,
                    "cara": cara,
                    "nama_cara": nama_met,
                    "N": N,
                    "dx_um": dx,
                    "phi_tot": phi_tot,
                    "phi_th": phi_th,
                    "Sv_um1": Sv,
                    "dt_per_dx": nisbah,
                    "ambang": ambang_cara,
                    "berkas_npy": nama_npy
                }
                baris_batu.append(row_entry)
                semua_baris_global.append(row_entry)
                print(f"    {ting['nama_tingkat']:<12} {nama_met:<27} {N:<6} {dx:<8.2f} {phi_tot:<10.4f} {phi_th:<10.4f} {Sv:<11.4f} {nisbah:<8.2f}", flush=True)

        # Simpan CSV per batuan
        jalur_csv_batu = os.path.join(dir_batuan, "tabel_multiresolusi.csv")
        with open(jalur_csv_batu, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["batuan", "tingkat", "cara", "nama_cara", "N", "dx_um", "phi_tot", "phi_th", "Sv_um1", "dt_per_dx", "ambang", "berkas_npy"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(baris_batu)

    if mode_t0 == "simulasi" and not skip_benchmark:
        print("\n>>> [BENCHMARK T0 - GPU] Menjalankan benchmark 100 langkah pada domain 256^3...")
        rock_bench = rocks[0]
        vol_bench = volume_cache[rock_bench][(0, "resolusi_penuh")]
        dur_100, mlups_100 = solver_lbm.benchmark_100_langkah(vol_bench, rock=rock_bench, arah="x")
        est_steps_total = sum(int(katalog.get(r, {}).get("langkah_acuan", {}).get("x", 15000)) * 3 for r in rocks)
        eta_detik = (dur_100 / 100.0) * est_steps_total
        print(f"    Kecepatan GPU CuPy  : {dur_100:.2f} detik / 100 langkah ({mlups_100:.1f} MLUPS)")
        print(f"    Estimasi Durasi T0  : ~{eta_detik:.1f} detik (~{eta_detik/60.0:.1f} menit) untuk {len(rocks)} batuan (x, y, z)", flush=True)

    for kunci in rocks:
        dir_batuan = dapatkan_subdirektori_batuan(out_dir, kunci)
        sim_dir = os.path.join(dir_batuan, "simulasi_multiresolusi")
        print(f"\n>>> [2/2] Eksekusi Pemecah Aliran LBM D3Q19 [GPU CuPy]: {kunci}")
        vol_t0 = volume_cache[kunci][(0, "resolusi_penuh")]

        for arah in ["x", "y", "z"]:
            out_f = os.path.join(sim_dir, f"{kunci}_T0_resolusi_penuh_{arah}.json")

            # 1. Prioritas Utama: Cari apakah SUDAH ADA hasil simulasi tunak nyata dari Langkah 0
            # Hanya hasil backend ini sendiri (hasil backend lain tidak disalin: komparasi CPU/GPU harus jujur)
            kandidat_jalur = [out_f]
            d_t0_tunak = None
            for cand_f in kandidat_jalur:
                if os.path.exists(cand_f):
                    try:
                        with open(cand_f, "r", encoding="utf-8") as f:
                            d_cand = json.load(f)
                        if d_cand.get("backend") in ["gpu", "cpu"] and tunak_sah(d_cand):
                            d_t0_tunak = d_cand
                            if cand_f != out_f:
                                with open(out_f, "w", encoding="utf-8") as f_out:
                                    json.dump(d_t0_tunak, f_out, indent=2)
                            break
                    except Exception:
                        pass

            if d_t0_tunak is not None:
                # Hasil simulasi tunak nyata sudah tersedia dari Langkah 0
                k_val = float(d_t0_tunak["K_mD"])
                step_val = int(d_t0_tunak["langkah"])
                print(f"    [{kunci}_T0_resolusi_penuh_{arah} | DARI LANGKAH 0] Menggunakan hasil tunak: K_mD = {k_val:.2f} mD ({step_val} langkah, konvergen=True)")
            else:
                # Belum ada T0 sah dari pemecah ini: SELALU disimulasikan, juga pada mode 'acuan'.
                # T0 data dosen tidak dipakai sebagai pembanding T1-T3 (pemecah dan kriteria berbeda).
                if mode_t0 == "acuan":
                    print(f"    [{kunci}_T0_resolusi_penuh_{arah}] T0 sah belum ada; disimulasikan (data acuan dosen tidak dipakai sebagai pembanding)")
                maks_t0 = dapatkan_batas_langkah_batuan(katalog, kunci, tingkat=0, arah=arah)
                solver_lbm.jalankan_kasus_lbm(
                    vol_t0, kunci, 0, "resolusi_penuh", arah,
                    out_file=out_f, maks_langkah=maks_t0, periksa=periksa
                )

        # Simulasi T1, T2, T3 (domain lebih kasar)
        for t_idx in [1, 2, 3]:
            for cara in ["majority", "block_avg"]:
                vol_t = volume_cache[kunci][(t_idx, cara)]
                for arah in ["x", "y", "z"]:
                    out_f = os.path.join(sim_dir, f"{kunci}_T{t_idx}_{cara}_{arah}.json")
                    maks_sub = dapatkan_batas_langkah_batuan(katalog, kunci, tingkat=t_idx, arah=arah)
                    if os.path.exists(out_f):
                        try:
                            with open(out_f, "r", encoding="utf-8") as f:
                                d_exist = json.load(f)
                            sama = abs(float(d_exist.get("phi_tot", -1)) - float(vol_t.mean())) < 1e-12
                            if sama and hasil_sah(d_exist) and (d_exist.get("konvergen") is True or d_exist.get("langkah", 0) >= maks_sub
                                         or d_exist.get("phi_th", 1.0) == 0.0):
                                continue
                        except Exception:
                            pass
                    solver_lbm.jalankan_kasus_lbm(
                        vol_t, kunci, t_idx, cara, arah,
                        out_file=out_f, maks_langkah=maks_sub, periksa=periksa
                    )

        print(f"    [OK] Berkas mandiri multiresolusi tersimpan di: {dir_batuan}/")

    # Jika memproses > 1 batuan, simpan juga tabel kompilasi di root out_dir
    if len(rocks) > 1:
        jalur_csv_global = os.path.join(out_dir, "tabel_multiresolusi.csv")
        with open(jalur_csv_global, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["batuan", "tingkat", "cara", "nama_cara", "N", "dx_um", "phi_tot", "phi_th", "Sv_um1", "dt_per_dx", "ambang", "berkas_npy"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(semua_baris_global)

    print("\n" + "=" * 80)
    print("SELESAI LANGKAH 05: Seluruh multiresolusi dan simulasi LBM D3Q19 selesai.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Langkah 05 [GPU]: Multiresolusi (T0-T3) & Simulasi LBM D3Q19")
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
