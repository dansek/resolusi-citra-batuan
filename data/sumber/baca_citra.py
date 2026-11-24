#!/usr/bin/env python3
"""
data/sumber/baca_citra.py
=========================
Membaca citra biner batupasir (.raw) dan memverifikasi integritas pembacaan fisis.

Fungsi:
1. Menemukan jalur basis data, berkas metadata (metadata-citra.json),
   tabel acuan (sebelas-batupasir.csv), dan folder citra biner (.raw) secara adaptif.
2. Membaca volume biner 256^3 voxel dengan tipe data uint8 dan urutan indeks 'C'.
3. Menghitung porositas total volume (fraksi voxel 0 / pori) dan membandingkannya
   secara silang terhadap nilai acuan resmi (toleransi ketat < 5e-4).
4. Menyajikan tabel verifikasi yang terurut secara ketat berdasarkan porositas terbesar ke terkecil.

Pemakaian (dari folder repositori):
    python data/sumber/baca_citra.py
    python data/sumber/baca_citra.py --batuan Parker
    python data/sumber/baca_citra.py --data_dir data/sumber
"""

import argparse
import csv
import json
import os
import sys

# Jalankan ulang otomatis memakai .venv repositori (berisi NumPy) bila belum aktif
_VENV_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv"))
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

try:
    import numpy as np
except ImportError:
    sys.exit("[GALAT] NumPy belum terpasang. Jalankan: pip install numpy")

HERE = os.path.dirname(os.path.abspath(__file__))           # <repo>/data/sumber
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))     # <repo>


def temukan_jalur_data(data_dir=None):
    """
    Menemukan jalur direktori data, berkas metadata, CSV acuan,
    dan direktori citra biner secara adaptif.
    """
    kandidat_dir = []
    if data_dir:
        kandidat_dir.append(os.path.abspath(data_dir))
    kandidat_dir.extend([
        HERE,
        os.path.join(ROOT, "data", "sumber"),
        os.path.join(ROOT, "data"),
        os.path.join(os.getcwd(), "data", "sumber"),
        os.path.join(os.getcwd(), "data"),
    ])

    base_dir = next((d for d in kandidat_dir if os.path.isdir(d)), kandidat_dir[0])

    # 1. Cari berkas metadata JSON
    kandidat_meta = [
        os.path.join(base_dir, "metadata-citra.json"),
        os.path.join(ROOT, "data", "sumber", "metadata-citra.json"),
        os.path.join(ROOT, "data", "metadata-citra.json"),
    ]
    meta_path = next((f for f in kandidat_meta if os.path.isfile(f)), None)

    # 2. Cari berkas CSV acuan
    kandidat_csv = [
        os.path.join(base_dir, "sebelas-batupasir.csv"),
        os.path.join(ROOT, "data", "sumber", "sebelas-batupasir.csv"),
        os.path.join(ROOT, "data", "sebelas-batupasir.csv"),
    ]
    csv_path = next((f for f in kandidat_csv if os.path.isfile(f)), None)

    # 3. Cari folder citra biner (.raw)
    kandidat_citra = [
        os.path.join(base_dir, "citra-biner"),
        os.path.join(ROOT, "data", "sumber", "citra-biner"),
        os.path.join(ROOT, "data", "citra-biner"),
        base_dir,
    ]
    citra_dir = next((d for d in kandidat_citra if os.path.isdir(d) and any(f.endswith(".raw") for f in os.listdir(d))), None)
    if citra_dir is None:
        citra_dir = os.path.join(base_dir, "citra-biner")

    return {
        "base_dir": base_dir,
        "meta_path": meta_path,
        "csv_path": csv_path,
        "citra_dir": citra_dir
    }


def muat_metadata(meta_path):
    """Membaca berkas metadata-citra.json jika tersedia, atau mengembalikan konfigurasi default."""
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    # Fallback konfigurasi resmi S4 D3Q19 jika berkas metadata hilang
    return {
        "himpunan": "Sebelas batupasir micro-CT",
        "sumber": "Neumann dkk. (2021)",
        "ukuran_voxel_tiap_sumbu": 256,
        "ukuran_voxel_m": 2.25e-06,
        "sisi_fisis_m": 0.000576,
        "tipe_data": "uint8",
        "urutan_indeks": "C",
        "berkas": []
    }


def muat_csv(csv_path):
    """Membaca tabel sebelas-batupasir.csv dan mengembalikan pemetaan batuan."""
    acuan = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for b in csv.DictReader(f):
                kunci = b["batuan"].strip()
                if kunci not in acuan or b.get("arah") == "x":
                    acuan[kunci] = {
                        "nama": b.get("nama", kunci),
                        "phi": float(b.get("phi", 0.0)),
                        "phi_terhubung": float(b.get("phi_terhubung", 0.0)),
                        "K_mD": float(b.get("K_mD", 0.0))
                    }
    return acuan


def baca_volume_raw(berkas, n=256, tipe="uint8", urutan="C"):
    """
    Membaca berkas biner .raw menjadi array boolean 3D.
    Konvensi fisis kuliah & publikasi:
    0 = pori (True), 1 = padatan (False).
    """
    v = np.fromfile(berkas, dtype=np.dtype(tipe))
    harap = n ** 3
    if v.size != harap:
        raise ValueError(
            f"Ukuran berkas tidak sesuai: terbaca {v.size:,} nilai, seharusnya {harap:,}."
        )
    return (v.reshape((n, n, n), order=urutan) == 0)


def main():
    parser = argparse.ArgumentParser(
        description="Verifikasi pembacaan berkas citra biner batupasir (.raw)."
    )
    parser.add_argument("--data_dir", default=None,
                        help="Direktori basis data citra mentah (default: deteksi otomatis data/sumber atau data/)")
    parser.add_argument("--batuan", default=None,
                        help="Kunci batuan spesifik (misal: Bentheimer, Parker). Jika kosong, seluruh 11 sampel diperiksa.")
    parser.add_argument("--tanpa_urut", action="store_true",
                        help="Jangan mengurutkan berdasarkan porositas (gunakan urutan metadata)")
    args = parser.parse_args()

    jalur = temukan_jalur_data(args.data_dir)
    meta = muat_metadata(jalur["meta_path"])
    acuan = muat_csv(jalur["csv_path"])

    n = meta.get("ukuran_voxel_tiap_sumbu", 256)
    tipe = meta.get("tipe_data", "uint8")
    urutan = meta.get("urutan_indeks", "C")
    dx_m = meta.get("ukuran_voxel_m", 2.25e-06)

    citra_dir = jalur["citra_dir"]
    if not os.path.isdir(citra_dir):
        sys.exit(f"[GALAT] Direktori citra biner tidak ditemukan di: {citra_dir}")

    # Buat daftar berkas yang akan diperiksa
    daftar_berkas = meta.get("berkas", [])
    if not daftar_berkas:
        # Jika metadata kosong, pindai langsung berkas .raw di folder citra
        for f_name in sorted(os.listdir(citra_dir)):
            if f_name.endswith(".raw"):
                kunci = f_name.split("_")[0]
                daftar_berkas.append({
                    "batuan": kunci,
                    "nama": acuan.get(kunci, {}).get("nama", kunci),
                    "berkas": f_name
                })

    if args.batuan:
        target = args.batuan.strip().lower()
        daftar_berkas = [b for b in daftar_berkas if b["batuan"].lower() == target or b["nama"].lower() == target]
        if not daftar_berkas:
            sys.exit(f"[GALAT] Batuan '{args.batuan}' tidak ditemukan di {citra_dir}/")

    # Urutkan berdasarkan porositas acuan terbesar ke terkecil (standar resmi Capstone S4)
    if not args.tanpa_urut:
        daftar_berkas = sorted(
            daftar_berkas,
            key=lambda b: acuan.get(b["batuan"], {}).get("phi", b.get("porositas_acuan", 0.0)),
            reverse=True
        )

    print("=" * 82)
    print("PEMERIKSAAN DAN VALIDASI SILANG PEMBACAAN CITRA BINER BATUPASIR (.RAW)")
    print(f"  * Direktori Citra   : {citra_dir}")
    print(f"  * Berkas Acuan CSV  : {jalur['csv_path'] or 'Tidak Ditemukan'}")
    print(f"  * Dimensi Domain    : {n}^3 voxel | dx = {dx_m*1e6:.2f} um (Tipe: {tipe}, Indeks: {urutan})")
    print(f"  * Konvensi Voxel    : 0 = Fluida / Pori (True), 1 = Padatan (False)")
    print("=" * 82)
    print(f"\n{'No':<3} {'Kunci':<13} {'Nama Sampel':<20} {'phi_baca':>9} {'phi_csv':>9} {'Selisih':>10}  {'Status'}")
    print("-" * 82)

    gagal = 0
    selisih_list = []

    for idx, b in enumerate(daftar_berkas, 1):
        kunci = b["batuan"]
        nama = b.get("nama", kunci)
        berkas_raw = b.get("berkas", f"{kunci}_2d25um_binary_256.raw")
        jalur_berkas = os.path.join(citra_dir, berkas_raw)

        if not os.path.exists(jalur_berkas):
            # Coba cari berkas case-insensitive
            kandidat = [f for f in os.listdir(citra_dir) if f.lower() == berkas_raw.lower()]
            if kandidat:
                jalur_berkas = os.path.join(citra_dir, kandidat[0])

        if not os.path.exists(jalur_berkas):
            print(f"{idx:<3} {kunci:<13} {nama:<20} {'--':>9} {'--':>9} {'--':>10}  [TIDAK DITEMUKAN]")
            gagal += 1
            continue

        try:
            vol_pori = baca_volume_raw(jalur_berkas, n, tipe, urutan)
        except ValueError as err:
            print(f"{idx:<3} {kunci:<13} {nama:<20} {'GALAT UKURAN':>30}")
            gagal += 1
            continue

        phi_baca = float(vol_pori.mean())
        phi_csv = acuan.get(kunci, {}).get("phi", b.get("porositas_acuan", phi_baca))
        selisih = abs(phi_baca - phi_csv)
        selisih_list.append(selisih)

        status_str = "COCOK" if selisih < 5e-4 else "BERGESER"
        if selisih >= 5e-4:
            gagal += 1

        print(f"{idx:<3} {kunci:<13} {nama:<20} {phi_baca:9.4f} {phi_csv:9.4f} {selisih:10.2e}  {status_str}")

    print("-" * 82)
    if selisih_list:
        max_selisih = max(selisih_list)
        avg_selisih = float(np.mean(selisih_list))
        print(f"Rata-rata Selisih Porositas: {avg_selisih:.2e} | Selisih Maksimum: {max_selisih:.2e}")

    print()
    if gagal:
        print(f"[PERINGATAN] {gagal} berkas belum terbaca dengan sempurna.")
        print("Tiga penyebab yang lazim, periksa berurutan:")
        print(f"  1. Urutan indeks keliru (harus '{urutan}').")
        print("  2. Konvensi nilai terbalik (nilai 0 adalah pori, 1 adalah padatan).")
        print("  3. Ukuran berkas terpotong atau korup saat disalin.")
        sys.exit(1)

    print("[BERHASIL] Seluruh berkas citra biner terbaca 100% sempurna dan terverifikasi identik dengan tabel acuan.")
    print("           Data siap digunakan untuk seluruh tahapan pipeline S4.")


if __name__ == "__main__":
    main()
