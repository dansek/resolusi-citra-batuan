#!/usr/bin/env python3
"""Ringkasan himpunan data sebelas batupasir.

Program ini membaca berkas data/sumber/sebelas-batupasir.csv, yaitu rekapitulasi hasil
simulasi Lattice Boltzmann D3Q19 pada sebelas batupasir berukuran 256^3 voxel
dengan ukuran voxel 2,25 mikrometer, lalu menghasilkan seluruh angka yang
tercantum pada Lampiran E.

Jalankan dari folder repositori:
    python data/sumber/sebelas_batupasir.py

Apabila berkas citra biner .raw tersedia, porositasnya dapat dihitung ulang
dengan pilihan --raw yang menunjuk folder berisi berkas tersebut (default: data/sumber/citra-biner).
"""

import argparse
import csv
import math
import os
import sys

# Jalankan ulang otomatis memakai .venv repositori (berisi NumPy) bila belum aktif
_VENV_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv"))
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

HERE =os.path.dirname(os.path.abspath(__file__))           # <repo>/data/sumber
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))     # <repo>


def cari_jalur_csv(jalur_input=None):
    """Mencari lokasi berkas sebelas-batupasir.csv secara adaptif."""
    if jalur_input and os.path.exists(jalur_input):
        return jalur_input
    kandidat = [
        os.path.join(HERE, "sebelas-batupasir.csv"),
        os.path.join(ROOT, "data", "sumber", "sebelas-batupasir.csv"),
        os.path.join(ROOT, "data", "sebelas-batupasir.csv"),
        os.path.join("data", "sumber", "sebelas-batupasir.csv"),
        os.path.join("data", "sebelas-batupasir.csv"),
    ]
    for k in kandidat:
        if os.path.exists(k):
            return k
    return kandidat[0]


def cari_jalur_raw(jalur_input=None):
    """Mencari folder citra biner mentah (.raw) secara adaptif."""
    if jalur_input and os.path.isdir(jalur_input):
        return jalur_input
    kandidat = [
        os.path.join(HERE, "citra-biner"),
        os.path.join(ROOT, "data", "sumber", "citra-biner"),
        os.path.join(ROOT, "data", "citra-biner"),
        os.path.join("data", "sumber", "citra-biner"),
        os.path.join("data", "citra-biner"),
        os.path.join(ROOT, "data", "sumber"),
        os.path.join("data", "sumber"),
    ]
    for k in kandidat:
        if os.path.isdir(k) and any(f.endswith(".raw") for f in os.listdir(k)):
            return k
    return None


def baca(berkas):
    """Mengembalikan dict nama batuan menuju dict arah menuju satu baris data."""
    data = {}
    with open(berkas, encoding="utf-8") as f:
        for b in csv.DictReader(f):
            data.setdefault(b["batuan"], {})[b["arah"]] = b
    return data


def rerata_geometrik(nilai):
    return math.exp(sum(math.log(v) for v in nilai) / len(nilai))


def cocokan_pangkat(phi, k):
    """Kuadrat terkecil atas log k = log a + m log phi. Mengembalikan m, a, R2."""
    x = [math.log10(p) for p in phi]
    y = [math.log10(v) for v in k]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    m = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    a = my - m * mx
    ss = sum((b - my) ** 2 for b in y)
    sr = sum((b - (a + m * p)) ** 2 for p, b in zip(x, y))
    return m, a, 1.0 - sr / ss


def porositas_dari_raw(folder, batuan, n=256):
    """Menghitung ulang porositas dari berkas biner, nilai 0 berarti pori."""
    import numpy as np

    if not folder or not os.path.isdir(folder):
        return None

    # Cocokkan awalan '<kunci>_' agar kunci pendek (mis. 'BB') tidak salah cocok
    pola = [f for f in os.listdir(folder)
            if f.lower().startswith(batuan.lower() + "_") and f.endswith(".raw")]
    if not pola:
        return None
    v = np.fromfile(os.path.join(folder, pola[0]), dtype=np.uint8)
    return float((v.reshape(n, n, n) == 0).mean())


LEBAR = 82
TOL_PHI = 5e-4


def judul_bagian(nomor, teks):
    print()
    print(f"[{nomor}] {teks}")
    print("-" * LEBAR)


def main():
    default_csv = cari_jalur_csv()
    default_raw = cari_jalur_raw()

    p = argparse.ArgumentParser(description="Ringkasan himpunan data sebelas batupasir.")
    p.add_argument("--csv", default=default_csv, help="Jalur berkas CSV sebelas-batupasir (default: %(default)s)")
    p.add_argument("--raw", default=default_raw, help="Folder berkas .raw untuk periksa ulang (default: %(default)s)")
    arg = p.parse_args()

    if not os.path.exists(arg.csv):
        sys.exit(f"[GALAT] Berkas CSV tidak ditemukan di: {arg.csv}")

    data = baca(arg.csv)
    urut = sorted(data, key=lambda b: -float(data[b]["x"]["phi"]))

    contoh = data[urut[0]]["x"]
    n_vox = int(contoh.get("n_voxel", 256))
    dx_um = float(contoh.get("dx_m", 2.25e-6)) * 1e6
    tau = contoh.get("tau_lb", "1.0")

    print("=" * LEBAR)
    print("RINGKASAN HIMPUNAN DATA SEBELAS BATUPASIR (LBM D3Q19 - LAMPIRAN E)")
    print(f"  * Berkas Acuan CSV  : {os.path.abspath(arg.csv)}")
    print(f"  * Folder Citra Biner: {os.path.abspath(arg.raw) if arg.raw else 'Tidak Ditemukan (validasi raw dilewati)'}")
    print(f"  * Dimensi Domain    : {n_vox}^3 voxel | dx = {dx_um:.2f} um | tau = {tau}")
    print(f"  * Jumlah Sampel     : {len(urut)} batuan (urut porositas terbesar ke terkecil)")
    print("=" * LEBAR)

    # [1] Tabel permeabilitas
    judul_bagian(1, "PERMEABILITAS TIGA ARAH & RERATA GEOMETRIK (mD)")
    print(f"{'No':<3} {'Nama Sampel':<20} {'phi':>7} {'phi_th':>7} {'Kx':>7} {'Ky':>7} {'Kz':>7} {'K_geo':>7} {'Aniso':>6}  {'Status'}")
    print("-" * LEBAR)
    phi_semua, kgeo_semua = [], []
    belum_tunak = 0
    for i, b in enumerate(urut, 1):
        r = data[b]
        k = [float(r[a]["K_mD"]) for a in "xyz"]
        kg = rerata_geometrik(k)
        tunak = all(int(r[a]["konvergen"]) for a in "xyz")
        belum_tunak += 0 if tunak else 1
        phi_semua.append(float(r["x"]["phi_terhubung"]))
        kgeo_semua.append(kg)
        print(f"{i:<3} {r['x']['nama']:<20} {float(r['x']['phi']):7.4f} {float(r['x']['phi_terhubung']):7.4f} "
              f"{k[0]:7.0f} {k[1]:7.0f} {k[2]:7.0f} {kg:7.0f} {max(k) / min(k):6.2f}  "
              f"{'TUNAK' if tunak else 'BELUM*'}")
    print("-" * LEBAR)
    print("Aniso = K_maks / K_min.")
    if belum_tunak:
        print(f"* {belum_tunak} batuan belum memenuhi patokan konvergensi pada 30.000 langkah.")

    # [2] Hukum pangkat
    m, a, r2 = cocokan_pangkat(phi_semua, kgeo_semua)
    meleset = [10 ** abs(math.log10(k) - (a + m * math.log10(p)))
               for p, k in zip(phi_semua, kgeo_semua)]
    khas = 10 ** (sum(math.log10(f) for f in meleset) / len(meleset))
    judul_bagian(2, "COCOKAN HUKUM PANGKAT  K_geo = a * phi_th^m")
    print(f"  * Eksponen m             : {m:.3f}")
    print(f"  * Konstanta log10(a)     : {a:.3f}")
    print(f"  * Koefisien R^2          : {r2:.4f}")
    print(f"  * Faktor meleset khas    : {khas:.2f} kali")
    print(f"  * Faktor meleset terbesar: {max(meleset):.2f} kali")

    # [3] Rentang permeabilitas
    semua_k = [float(data[b][a2]["K_mD"]) for b in data for a2 in "xyz"]
    judul_bagian(3, "RENTANG PERMEABILITAS SELURUH ARAH")
    print(f"  * K minimum              : {min(semua_k):.0f} mD")
    print(f"  * K maksimum             : {max(semua_k):.0f} mD")
    print(f"  * Rasio maks / min       : {max(semua_k) / min(semua_k):.0f} kali")

    # [4] Pori terisolasi
    judul_bagian(4, "FRAKSI RUANG PORI YANG TIDAK TERSAMBUNG")
    print(f"{'No':<3} {'Nama Sampel':<20} {'phi':>9} {'phi_th':>9} {'Terisolasi':>12}")
    print("-" * LEBAR)
    for i, b in enumerate(urut, 1):
        r = data[b]["x"]
        phi, phi_th = float(r["phi"]), float(r["phi_terhubung"])
        print(f"{i:<3} {r['nama']:<20} {phi:9.4f} {phi_th:9.4f} {100 * (1.0 - phi_th / phi):10.2f} %")
    print("-" * LEBAR)

    # [5] Validasi silang dengan berkas raw
    if not arg.raw:
        print()
        print("[INFO] Folder citra biner (.raw) tidak ditemukan; validasi silang porositas dilewati.")
        return

    judul_bagian(5, "VALIDASI SILANG POROSITAS DARI BERKAS BINER (.RAW)")
    print(f"{'No':<3} {'Nama Sampel':<20} {'phi_csv':>9} {'phi_raw':>9} {'Selisih':>10}  {'Status'}")
    print("-" * LEBAR)
    gagal = 0
    selisih_list = []
    for i, b in enumerate(urut, 1):
        nama = data[b]["x"]["nama"]
        tercatat = float(data[b]["x"]["phi"])
        phi = porositas_dari_raw(arg.raw, b, n=n_vox)
        if phi is None:
            print(f"{i:<3} {nama:<20} {tercatat:9.4f} {'--':>9} {'--':>10}  [TIDAK DITEMUKAN]")
            gagal += 1
            continue
        selisih = abs(phi - tercatat)
        selisih_list.append(selisih)
        cocok = selisih < TOL_PHI
        gagal += 0 if cocok else 1
        print(f"{i:<3} {nama:<20} {tercatat:9.4f} {phi:9.4f} {selisih:10.2e}  {'COCOK' if cocok else 'BERGESER'}")
    print("-" * LEBAR)
    if selisih_list:
        print(f"Rata-rata Selisih Porositas: {sum(selisih_list) / len(selisih_list):.2e} | "
              f"Selisih Maksimum: {max(selisih_list):.2e}")

    print()
    if gagal:
        print(f"[PERINGATAN] {gagal} batuan tidak cocok atau berkas .raw tidak ditemukan.")
        sys.exit(1)
    print("[BERHASIL] Porositas seluruh berkas citra biner identik dengan tabel acuan CSV.")


if __name__ == "__main__":
    main()
