#!/usr/bin/env python3
"""
program/run.py — Launcher Interaktif Terpadu Capstone Project S4
========================================================
Topik S4: "Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)"
Mata Kuliah: Simulasi Berbasis Data Citra (Magister Inovasi Regional, Unpad)

Fitur Utama:
1. Deteksi perangkat keras otomatis:
   - CPU (model prosesor & jumlah core/thread)
   - GPU NVIDIA (ketersediaan CUDA/CuPy, nama GPU, dan pemeriksaan VRAM >= 4.0 GB)
2. Menu CLI Interaktif (5 tahapan konfigurasi):
   (1) Pilihan Backend Komputasi: GPU (`program/gpu/`) atau CPU (`program/cpu/`)
   (2) Mode Eksekusi: "Run-All" (Langkah 00 s.d. 07 berurutan) atau "Per-Langkah" (pilih langkah 0-7)
   (3) Folder Output Penyimpanan Hasil (default: `hasil/`)
   (4) Pilihan Sampel Batuan Pasir:
       1. Sampel Utama Capstone S4 (Bentheimer & BSG)
       2. Seluruh 11 Batuan Pasir (Scan & Batch Run)
       3. Pilih Batuan Tertentu (daftar dinamis 11 batuan)
   (5) Mode Konvergensi Resolusi Penuh T0 (256^3):
       - Opsi A: Menggunakan data konvergensi acuan resmi (instan & tervalidasi).
       - Opsi B: Simulasi ulang konvergensi dari awal dengan interval pemeriksaan (default 500 langkah),
                 didahului benchmark singkat 100 langkah untuk menampilkan estimasi durasi (ETA)
                 pada perangkat keras pengguna sebelum simulasi penuh berjalan.
3. Struktur Penyimpanan Mandiri:
   - Setiap batuan memiliki subdirektori tersendiri di `hasil/<NamaBatuan>/`
   - Ringkasan komparasi lintas batuan dibuat di `hasil/` jika memproses lebih dari 1 batuan.
4. Mendukung eksekusi non-interaktif melalui argumen CLI untuk otomasi/pengujian.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # akar repositori (run.py ada di program/)
_VENV_DIR = os.path.join(ROOT, ".venv")
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import argparse
import platform
import subprocess
import importlib.util
import inspect
import numpy as np
from program.katalog_batuan import muat_semua_batuan, urutkan_daftar_batuan, dapatkan_subdirektori_batuan, dapatkan_batas_langkah_batuan
from program.pemuat_modul import muat as _muat
from program.utilitas_hardware import deteksi_topologi_cpu, terapkan_afinitas_core_fisik
from program.unduh_data import pastikan_citra, muat_metadata

DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil")

DAFTAR_LANGKAH = {
    0: ("00_verifikasi_pemecah.py", "Langkah 00: Verifikasi Pemecah terhadap Solusi Stokes Analitik (luaran 5)"),
    1: ("01_permeabilitas_resolusi_penuh.py", "Langkah 01: Permeabilitas Kondisi Tunak Resolusi Penuh T0 (luaran 4)"),
    2: ("02_rentang_segmentasi.py", "Langkah 02: Rentang Segmentasi, LBM pada 7 Geseran Antarmuka (luaran 2, 7)"),
    3: ("03_sebaran_leher_pori.py", "Langkah 03: Sebaran Leher Pori d_t dan Kelayakan Citra (luaran 1)"),
    4: ("04_volume_elementer.py", "Langkah 04: Volume Elementer Representatif, Uji Ukuran Bertingkat (luaran 3)"),
    5: ("05_multiresolusi_lbm.py", "Langkah 05: Penurunan Resolusi T0-T3 Dua Cara + LBM D3Q19 (luaran 10)"),
    6: ("06_evaluasi_kelayakan_resolusi.py", "Langkah 06: Kelayakan, Simpangan, Tingkat Kritis, Medan Pandang (luaran 9, 11, 12)"),
    7: ("07_rekapitulasi_visualisasi.py", "Langkah 07: Rekapitulasi Tiga Sifat dan Grafik Akhir (luaran 6, 13)"),
    8: ("08_komparasi_cpu_gpu.py", "Langkah 08: Komparasi GPU vs CPU, Waktu dan Memori (durasi produksi, perangkat menganggur)"),
    9: (None, "Langkah 09: Angka Turunan dan Gambar Laporan (hasil/angka_laporan.csv, gambar/)"),
}
BATUAN_LAPORAN = ("Bentheimer", "BSG")


def angka_dan_gambar_laporan(backend, out_dir, base_out_dir, rocks):
    """Langkah 09: angka turunan dan gambar laporan dari hasil backend ini, tanpa simulasi.
    Hanya untuk batuan laporan (Bentheimer dan BSG); dilewati pada mode uji agar gambar laporan tidak tertimpa."""
    if os.environ.get("S4_UJI"):
        print("    [LEWAT] Mode uji: angka dan gambar laporan tidak dibuat ulang.")
        return
    tersedia = [b for b in BATUAN_LAPORAN if os.path.exists(os.path.join(out_dir, b, "rekapitulasi_akhir_s4.json"))]
    if len(tersedia) < len(BATUAN_LAPORAN):
        print(f"    [LEWAT] Perlu hasil lengkap {' dan '.join(BATUAN_LAPORAN)} di {out_dir} (batuan dijalankan: {', '.join(rocks)}).")
        return
    from program import angka_laporan, buat_gambar
    angka_laporan.main(base_out_dir, backend)
    gambar = os.path.join(ROOT, "gambar") if os.path.abspath(base_out_dir) == DEFAULT_OUT_DIR else os.path.join(base_out_dir, "gambar")
    buat_gambar.main(out_dir, gambar)


def deteksi_perangkat_keras():
    """Mendeteksi spesifikasi CPU (Core Fisik vs SMT) dan ketersediaan GPU NVIDIA (CUDA & VRAM >= 4 GB)."""
    top_cpu = deteksi_topologi_cpu()

    gpu_info = {
        "tersedia": False,
        "layak_vram": False,
        "nama": "Tidak terdeteksi",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "catatan": "CuPy / CUDA tidak ditemukan"
    }

    try:
        import cupy as cp
        if cp.cuda.is_available():
            props = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = props.get("name", b"NVIDIA GPU")
            nama_gpu = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            vram_total_gb = mem_total / (1024 ** 3)
            vram_free_gb = mem_free / (1024 ** 3)
            layak = (vram_total_gb >= 4.0)

            gpu_info = {
                "tersedia": True,
                "layak_vram": layak,
                "nama": nama_gpu,
                "vram_total_gb": vram_total_gb,
                "vram_free_gb": vram_free_gb,
                "catatan": "Siap (VRAM >= 4 GB)" if layak else "VRAM < 4 GB (Disarankan mode CPU)"
            }
    except Exception as e:
        gpu_info["catatan"] = f"Tidak aktif ({e})"

    return {
        "cpu_topologi": top_cpu,
        "cpu_cores": top_cpu["core_fisik"],
        "cpu_threads": top_cpu["total_threads"],
        "cpu_nama": top_cpu["model"],
        "gpu": gpu_info
    }


def muat_modul_langkah(backend, nama_skrip):
    """Memuat modul skrip dari folder program/gpu atau program/cpu secara dinamis."""
    jalur_skrip = os.path.join(ROOT, "program", backend, f"{backend}_{nama_skrip}")
    if not os.path.exists(jalur_skrip):
        raise FileNotFoundError(f"Skrip tidak ditemukan di: {jalur_skrip}")
    nama_mod = f"mod_{backend}_{os.path.splitext(nama_skrip)[0]}"
    spec = importlib.util.spec_from_file_location(nama_mod, jalur_skrip)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jalankan_benchmark_t0(backend, rocks):
    """Menjalankan 100 langkah LBM D3Q19 resolusi penuh 256^3 untuk mengukur ETA aktual."""
    print("\n" + "-" * 78)
    print(f"[BENCHMARK HARDWARE - {backend.upper()}] Mengukur kecepatan LBM D3Q19 (100 langkah @ 256^3)...")
    print("-" * 78, flush=True)

    kat = muat_semua_batuan(DATA_DIR)
    solver_mod = muat_modul_langkah(backend, "solver_lbm.py")
    rock_uji = rocks[0]
    info_uji = kat.get(rock_uji, {})
    raw_name = info_uji.get("berkas", f"{rock_uji}_2d25um_binary_256.raw")
    jalur_raw = info_uji.get("jalur_berkas")
    if not jalur_raw or not os.path.exists(jalur_raw):
        jalur_raw = os.path.join(DATA_DIR, "sumber", "citra-biner", raw_name)
    if not os.path.exists(jalur_raw):
        jalur_raw = os.path.join(DATA_DIR, "citra-biner", raw_name)

    v = np.fromfile(jalur_raw, dtype=np.uint8)
    pori = (v.reshape((256, 256, 256)) == 0)

    dur_100, mlups = solver_mod.benchmark_100_langkah(pori, rock=rock_uji, arah="x")

    est_steps_t0 = 0
    for r in rocks:
        for arah in ["x", "y", "z"]:
            est_steps_t0 += dapatkan_batas_langkah_batuan(kat, r, tingkat=0, arah=arah)

    eta_t0_detik = (dur_100 / 100.0) * est_steps_t0
    eta_t0_menit = eta_t0_detik / 60.0

    print(f"  * Sampel Benchmark         : {rock_uji} (256^3 voxel, domain LBM [264, 258, 258])")
    print(f"  * Durasi 100 Langkah       : {dur_100:.2f} detik ({mlups:.1f} MLUPS)")
    print(f"  * Estimasi Total Langkah T0: ~{est_steps_t0:,} langkah ({len(rocks)} batuan, arah x, y, z)")
    print(f"  * Estimasi Waktu Selesai   : ~{eta_t0_detik:.1f} detik (~{eta_t0_menit:.1f} menit)")
    print("-" * 78, flush=True)
    return dur_100, mlups, eta_t0_detik


def tanya_pilihan(prompt, opsi_valid, default_val):
    """Membaca input pengguna secara aman dengan nilai default."""
    while True:
        try:
            ans = input(f"{prompt} [default: {default_val}]: ").strip()
        except EOFError:
            return default_val
        if not ans:
            return default_val
        if ans in opsi_valid:
            return ans
        print(f"  [!] Pilihan '{ans}' tidak valid. Pilih salah satu dari: {', '.join(opsi_valid)}")


def interaksi_menu():
    """Mengelola antarmuka CLI interaktif 5 langkah."""
    hw = deteksi_perangkat_keras()
    kat = muat_semua_batuan(DATA_DIR)
    # Daftar batuan diambil dari metadata, sehingga batuan yang citranya belum diunduh tetap dapat dipilih
    meta_b = {b["batuan"]: b for b in muat_metadata()["berkas"]}
    daftar_urut = sorted(meta_b, key=lambda k: meta_b[k]["porositas_acuan"], reverse=True)

    print("=" * 80)
    print("  CAPSTONE PROJECT S4 — BATAS KEKASARAN CITRA BATUAN PASIR (LBM D3Q19)")
    print("  Mata Kuliah Simulasi Berbasis Data Citra | Magister Inovasi Regional Unpad")
    print("=" * 80)
    print("\n[DETEKSI PERANGKAT KERAS OTOMATIS]")
    top_cpu = hw.get("cpu_topologi", {})
    smt_str = f" | SMT: {top_cpu.get('threads_per_core', 2)} Thread/Core" if top_cpu.get("smt_aktif") else ""
    print(f"  * CPU Terdeteksi : {hw['cpu_nama']}")
    print(f"                     {hw['cpu_cores']} Core Fisik Murni ({hw['cpu_threads']} Thread Logis Total{smt_str})")
    g = hw["gpu"]
    if g["tersedia"]:
        print(f"  * GPU Terdeteksi : {g['nama']} | VRAM Total: {g['vram_total_gb']:.2f} GB (Bebas: {g['vram_free_gb']:.2f} GB)")
        print(f"  * Status GPU     : {g['catatan']}")
    else:
        print(f"  * GPU Terdeteksi : Tidak tersedia ({g['catatan']})")

    # 1. Pilihan CPU atau GPU
    default_backend = "1" if (g["tersedia"] and g["layak_vram"]) else "2"
    print("\n[1] PILIH BACKEND KOMPUTASI:")
    print("    1. GPU (CuPy / NVIDIA CUDA - Akselerasi Paralel VRAM >= 4GB)")
    print("    2. CPU (NumPy / SciPy / Numba - Multi-Core JIT Paralel)")
    p_back = tanya_pilihan("    Masukkan pilihan (1/2)", ["1", "2"], default_backend)
    if p_back == "1" and not (g["tersedia"] and g["layak_vram"]):
        print("    [PERINGATAN] GPU CUDA dengan VRAM >= 4GB tidak terdeteksi! Beralih otomatis ke mode CPU.")
        backend = "cpu"
    else:
        backend = "gpu" if p_back == "1" else "cpu"

    # 2. Mode Run-All atau Per-Langkah
    print("\n[2] PILIH MODE EKSEKUSI PIPELINE:")
    print("    1. Run-All (Jalankan Langkah 00 s.d. 07 secara berurutan)")
    print("    2. Per-Langkah (Pilih langkah spesifik 0 s.d. 7)")
    p_mode = tanya_pilihan("    Masukkan pilihan (1/2)", ["1", "2"], "1")
    if p_mode == "1":
        langkah_dipilih = sorted(DAFTAR_LANGKAH)
    else:
        print("\n    Daftar Langkah Pipeline:")
        for k, (_, desk) in DAFTAR_LANGKAH.items():
            print(f"      [{k}] {desk}")
        while True:
            try:
                raw_s = input("    Masukkan nomor langkah (misal: 1 atau 0,1,2,3,4,5,6,7) [default: 0]: ").strip()
            except EOFError:
                raw_s = "0"
            if not raw_s:
                raw_s = "0"
            try:
                parts = [int(x.strip()) for x in raw_s.replace(" ", ",").split(",") if x.strip() != ""]
                if parts and all(p in DAFTAR_LANGKAH for p in parts):
                    langkah_dipilih = sorted(list(set(parts)))
                    break
            except ValueError:
                pass
            print("    [!] Masukkan angka langkah antara 0 s.d. 8.")

    # 3. Folder Output
    print("\n[3] FOLDER OUTPUT PENYIMPANAN HASIL:")
    try:
        in_out = input("    Masukkan nama/path folder output [default: hasil]: ").strip()
    except EOFError:
        in_out = ""
    out_dir = os.path.abspath(in_out) if in_out else DEFAULT_OUT_DIR

    # 4. Pilihan Batuan Pasir
    print("\n[4] PILIH SAMPEL BATUAN PASIR:")
    print("    1. Sampel Utama Capstone S4 (Bentheimer & BSG)")
    print(f"    2. Seluruh {len(daftar_urut)} Batuan Pasir (Scan & Batch Run)")
    print(f"    3. Pilih Batuan Tertentu (Pilih dari daftar {len(daftar_urut)} batuan)")
    p_rock = tanya_pilihan("    Masukkan pilihan (1/2/3)", ["1", "2", "3"], "1")

    if p_rock == "1":
        rocks = ["Bentheimer", "BSG"]
    elif p_rock == "2":
        rocks = list(daftar_urut)
    else:
        print("\n    Daftar Sampel Batuan Pasir Tersedia:")
        for idx, k in enumerate(daftar_urut, 1):
            if k in kat:
                b_info = kat[k]
                print(f"      [{idx:2d}] {k:<13} - {b_info['nama']} (phi={b_info['phi_acuan']:.4f}, K~{b_info['K_geo_acuan']:.1f} mD)")
            else:
                print(f"      [{idx:2d}] {k:<13} - {meta_b[k]['nama']} (phi={meta_b[k]['porositas_acuan']:.4f}, belum diunduh)")
        while True:
            try:
                raw_s = input(f"    Masukkan nomor batuan (misal: 1 atau 1,2,6) [default: 1]: ").strip()
            except EOFError:
                raw_s = "1"
            if not raw_s:
                raw_s = "1"
            try:
                nomor_parts = [int(x.strip()) for x in raw_s.replace(" ", ",").split(",") if x.strip() != ""]
                if nomor_parts and all(1 <= p <= len(daftar_urut) for p in nomor_parts):
                    nomor_parts = sorted(list(set(nomor_parts)))
                    rocks = [daftar_urut[p - 1] for p in nomor_parts]
                    break
            except ValueError:
                pass
            print(f"    [!] Masukkan nomor batuan antara 1 s.d. {len(daftar_urut)}.")

    # Unduh hanya citra batuan yang dipilih (yang sudah ada dan sidiknya cocok dilewati)
    print("\n    Memeriksa citra batuan terpilih:")
    belum = pastikan_citra(rocks, DATA_DIR)
    if belum:
        sys.exit(f"[GALAT] Citra belum tersedia: {', '.join(belum)}. Periksa koneksi atau lihat README bagian Persiapan.")

    # 5. Mode Konvergensi Resolusi Penuh T0 (256^3)
    if 1 not in langkah_dipilih and 5 not in langkah_dipilih:
        mode_t0 = "acuan"
        periksa = 500
        skip_bench_in_step3 = False
    else:
        print("\n[5] MODE KONVERGENSI RESOLUSI PENUH T0 (256^3):")
        print("    A. Gunakan hasil Langkah 0 / Data Acuan yang tersedia (Instan & Direkomendasikan)")
        print("    B. Simulasi ulang konvergensi T0 dari awal (dengan Benchmark ETA & Pemeriksaan)")
        p_t0 = tanya_pilihan("    Masukkan pilihan (A/B)", ["A", "B", "a", "b"], "A").upper()
        mode_t0 = "acuan" if p_t0 == "A" else "simulasi"
        periksa = 500
        skip_bench_in_step3 = False

        if mode_t0 == "simulasi":
            try:
                in_p = input("    Masukkan interval langkah pemeriksaan konvergensi [default: 500]: ").strip()
                if in_p:
                    periksa = max(50, int(in_p))
            except (EOFError, ValueError):
                periksa = 500

            jalankan_benchmark_t0(backend, rocks)
            skip_bench_in_step3 = True
            konfirmasi = tanya_pilihan("    Lanjutkan eksekusi simulasi penuh T0 sekarang? (Y/N)", ["Y", "N", "y", "n"], "Y").upper()
            if konfirmasi == "N":
                print("\n    [INFO] Eksekusi dibatalkan oleh pengguna. Program dihentikan.")
                sys.exit(0)

    return {
        "backend": backend,
        "langkah": langkah_dipilih,
        "out_dir": out_dir,
        "rocks": rocks,
        "mode_t0": mode_t0,
        "periksa": periksa,
        "skip_benchmark": skip_bench_in_step3
    }


def eksekusi_pipeline(konfig):
    backend = konfig["backend"]
    langkah_list = konfig["langkah"]
    raw_out = konfig["out_dir"]
    rocks = konfig["rocks"]
    mode_t0 = konfig["mode_t0"]
    periksa = konfig["periksa"]
    skip_benchmark = konfig.get("skip_benchmark", False)

    # Pisahkan folder output berdasarkan backend (misal: hasil/gpu atau hasil/cpu)
    if os.path.basename(raw_out) in ["gpu", "cpu"]:
        out_dir = raw_out
        base_out_dir = os.path.dirname(raw_out)
    else:
        out_dir = os.path.join(raw_out, backend)
        base_out_dir = raw_out
    os.makedirs(out_dir, exist_ok=True)

    if backend == "cpu":
        terapkan_afinitas_core_fisik(verbose=True)

    print("\n" + "=" * 80)
    print("RINGKASAN KONFIGURASI EKSEKUSI:")
    print(f"  * Backend Komputasi : {backend.upper()} (program/{backend}/)")
    print(f"  * Langkah Dijalankan: {', '.join(f'{k:02d}' for k in langkah_list)}")
    print(f"  * Sampel Batuan     : {', '.join(rocks)}")
    print(f"  * Mode Konvergensi  : {'Opsi A (Data Konvergensi Acuan Instan)' if mode_t0 == 'acuan' else f'Opsi B (Simulasi Penuh T0, Pemeriksaan={periksa} step)'}")
    print(f"  * Folder Induk      : {base_out_dir}")
    print(f"  * Direktori Backend : {out_dir}")
    print("=" * 80, flush=True)

    t_awal_total = time.monotonic()

    for nomor in langkah_list:
        nama_file, judul = DAFTAR_LANGKAH[nomor]
        print(f"\n>>> Menjalankan {judul}...")
        t_step = time.monotonic()
        if nomor == 9:
            angka_dan_gambar_laporan(backend, out_dir, base_out_dir, rocks)
            print(f"<<< Selesai Langkah {nomor:02d} dalam {time.monotonic() - t_step:.2f} detik.", flush=True)
            continue
        mod = muat_modul_langkah(backend, nama_file)

        argumen = dict(rocks=rocks, data_dir=konfig.get("data_dir", DATA_DIR), out_dir=out_dir, mode_t0=mode_t0,
                       periksa=periksa, skip_benchmark=skip_benchmark)
        diterima = inspect.signature(mod.jalankan).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in diterima.values()):
            argumen = {k: v for k, v in argumen.items() if k in diterima}
        mod.jalankan(**argumen)
        print(f"<<< Selesai Langkah {nomor:02d} dalam {time.monotonic() - t_step:.2f} detik.", flush=True)

    dur_total = time.monotonic() - t_awal_total
    print("\n" + "=" * 80)
    print(f"SELURUH PIPELINE SELESAI DENGAN SUKSES! (Total Durasi: {dur_total:.2f} detik)")
    print(f"Seluruh tabel CSV, ringkasan JSON, dan grafik ilmiah tersimpan mandiri di: {out_dir}/")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Launcher Terpadu Capstone Project S4: Batas Kekasaran Citra")
    parser.add_argument("--non-interactive", action="store_true", help="Jalankan langsung tanpa prompt menu CLI")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None, help="Pilih backend komputasi (cpu/gpu)")
    parser.add_argument("--steps", nargs="+", type=int, default=sorted(DAFTAR_LANGKAH), choices=sorted(DAFTAR_LANGKAH),
                        help="Daftar nomor langkah pipeline yang akan dijalankan (default: 0 1 2 3 4 5 6 7 8)")
    parser.add_argument("--rocks", nargs="+", default=["Bentheimer", "BSG"],
                        help="Daftar sampel batuan atau 'all' untuk seluruh 11 batuan")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Direktori basis data citra biner (.raw)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Direktori penyimpanan hasil")
    parser.add_argument("--mode_t0", choices=["acuan", "simulasi"], default="acuan",
                        help="Mode konvergensi T0: 'acuan' (instan) atau 'simulasi' (LBM penuh)")
    parser.add_argument("--periksa", type=int, default=500,
                        help="Interval langkah pemeriksaan konvergensi untuk mode simulasi")
    parser.add_argument("--skip_benchmark", action="store_true", help="Lewati benchmark 100 langkah")
    parser.add_argument("--tanpa_unduh", action="store_true",
                        help="Jangan mengunduh citra dari Google Drive; hanya periksa sidik berkas lokal")
    parser.add_argument("--uji", type=int, nargs="?", const=128, default=None, metavar="SISI",
                        help="Mode uji cepat (fase perbaikan): citra dipotong SISI^3 di tengah (bawaan 128), "
                             "LBM dibatasi --uji_langkah langkah, keluaran ke hasil_uji/ dan ditandai 'uji'")
    parser.add_argument("--uji_langkah", type=int, default=2000, help="Batas langkah LBM dalam mode uji")
    args = parser.parse_args()

    if args.uji is not None:
        if args.uji % 8 or args.uji < 64:
            sys.exit("[GALAT] --uji harus kelipatan 8 dan >= 64 (T3 = SISI/8).")
        os.environ["S4_UJI"] = str(args.uji)
        os.environ["S4_UJI_LANGKAH"] = str(args.uji_langkah)
        if os.path.abspath(args.out_dir) == os.path.abspath(DEFAULT_OUT_DIR):
            args.out_dir = os.path.join(ROOT, "hasil_uji")
        args.mode_t0 = "simulasi"          # acuan dosen berukuran 256^3, tidak berlaku untuk potongan
        args.skip_benchmark = True
        print(f"[MODE UJI] potongan {args.uji}^3, maks {args.uji_langkah} langkah, keluaran {args.out_dir} "
              "(BUKAN hasil laporan)", flush=True)

    if not args.non_interactive:
        konfig = interaksi_menu()
    else:
        if args.backend is None:
            hw = deteksi_perangkat_keras()      # membuka konteks CUDA; hanya bila backend belum ditentukan
            backend = "gpu" if (hw["gpu"]["tersedia"] and hw["gpu"]["layak_vram"]) else "cpu"
        else:
            backend = args.backend              # --backend cpu tidak menyentuh GPU sama sekali

        rocks_arg = args.rocks
        # Unduh hanya citra batuan yang diminta; 'all' mengunduh seluruh batuan di metadata
        print("[DATA] Memeriksa citra batuan:", flush=True)
        belum = pastikan_citra(rocks_arg, args.data_dir, unduh=not args.tanpa_unduh)
        if belum:
            sys.exit(f"[GALAT] Citra belum tersedia: {', '.join(belum)}. Unduh dengan program/unduh_data.py atau letakkan "
                     f"berkasnya di {os.path.join(os.path.abspath(args.data_dir), 'sumber', 'citra-biner')}/ "
                     "(lihat README bagian Persiapan).")
        kat = muat_semua_batuan(args.data_dir)
        if len(rocks_arg) == 1 and rocks_arg[0].lower() in ["all", "semua"]:
            rocks_arg = urutkan_daftar_batuan(kat)
        tak_dikenal = [r for r in rocks_arg if r not in kat]
        if tak_dikenal:
            sys.exit(f"[GALAT] Batuan tidak dikenal: {', '.join(tak_dikenal)}. Pilihan: {', '.join(urutkan_daftar_batuan(kat))}")

        konfig = {
            "backend": backend,
            "langkah": sorted(list(set(args.steps))),
            "out_dir": os.path.abspath(args.out_dir),
            "data_dir": os.path.abspath(args.data_dir),
            "rocks": rocks_arg,
            "mode_t0": args.mode_t0,
            "periksa": args.periksa,
            "skip_benchmark": args.skip_benchmark
        }

    eksekusi_pipeline(konfig)


if __name__ == "__main__":
    main()
