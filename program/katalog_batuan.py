"""
program/katalog_batuan.py
=========================
Modul Manajemen Katalog Sampel Batuan Pasir (Dinamis & Mandiri)
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Modul ini memindai citra biner mentah (.raw) dan metadata acuan (sebelas-batupasir.csv),
serta menyediakan data konfigurasi untuk setiap batuan secara dinamis.
"""

import os
import csv
import glob
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")

NAMA_LENGKAP_DEFAULT = {
    "Bentheimer": ("Bentheimer Sandstone", "Porositas Tinggi & Permeabilitas Besar"),
    "Parker": ("Parker Sandstone", "Batuan Ketat / Porositas Rendah"),
    "BanderaBrown": ("Bandera Brown Sandstone", "Batuan Ketat / Porositas Sedang"),
    "BanderaGray": ("Bandera Gray Sandstone", "Batuan Ketat / Porositas Sedang"),
    "BB": ("Buff Berea Sandstone", "Porositas Sedang-Tinggi"),
    "Berea": ("Berea Sandstone", "Batuan Acuan Standar Industri"),
    "BSG": ("Berea Sister Gray Sandstone", "Porositas Sedang"),
    "BUG": ("Berea Upper Gray Sandstone", "Porositas Sedang"),
    "CastleGate": ("Castlegate Sandstone", "Porositas Tinggi & Permeabilitas Besar"),
    "Kirby": ("Kirby Sandstone", "Porositas Sedang"),
    "Leopard": ("Leopard Sandstone", "Porositas Sedang & Permeabilitas Tinggi")
}


def cari_jalur_data(data_dir=None):
    """Mencari lokasi folder data yang valid."""
    kandidat = []
    if data_dir:
        kandidat.append(data_dir)
        kandidat.append(os.path.join(data_dir, "sumber"))
    kandidat.extend([
        os.path.join(ROOT, "data"),
        os.path.join(ROOT, "data", "sumber"),
        os.path.join(HERE, "..", "data"),
        os.path.join(HERE, "..", "data", "sumber")
    ])
    for p in kandidat:
        if os.path.isdir(p):
            # Periksa apakah ada citra-biner atau sebelas-batupasir.csv
            if os.path.isdir(os.path.join(p, "citra-biner")) or os.path.exists(os.path.join(p, "sebelas-batupasir.csv")):
                return os.path.abspath(p)
            if os.path.isdir(os.path.join(p, "sumber", "citra-biner")):
                return os.path.abspath(os.path.join(p, "sumber"))
    return os.path.abspath(DATA_DIR)


def muat_semua_batuan(data_dir=None):
    """
    Memindai seluruh berkas raw batuan dan memuat metadatanya.
    Mengembalikan dict: { 'Bentheimer': { ... }, 'Parker': { ... }, ... }
    """
    base_data = cari_jalur_data(data_dir)
    dir_biner = os.path.join(base_data, "citra-biner")
    if not os.path.isdir(dir_biner):
        # Coba periksa data/sumber/citra-biner
        sub = os.path.join(base_data, "sumber", "citra-biner")
        if os.path.isdir(sub):
            dir_biner = sub

    # Baca metadata acuan CSV jika ada
    jalur_csv = os.path.join(base_data, "sebelas-batupasir.csv")
    if not os.path.exists(jalur_csv):
        jalur_csv = os.path.join(base_data, "sumber", "sebelas-batupasir.csv")

    metadata_csv = {}
    if os.path.exists(jalur_csv):
        try:
            with open(jalur_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    b_kunci = r.get("batuan", "").strip()
                    arah = r.get("arah", "").strip()
                    if b_kunci:
                        metadata_csv.setdefault(b_kunci, {})[arah] = r
        except Exception:
            pass

    # Pindai file raw
    katalog = {}
    pola = os.path.join(dir_biner, "*_2d25um_binary_256.raw")
    berkas_raw = sorted(glob.glob(pola))

    for jalur_file in berkas_raw:
        nama_file = os.path.basename(jalur_file)
        kunci = nama_file.split("_")[0]

        nama_default, tipe_default = NAMA_LENGKAP_DEFAULT.get(kunci, (f"{kunci} Sandstone", "Batupasir"))

        csv_info = metadata_csv.get(kunci, {})
        phi_total = None
        phi_terhubung = None
        k_arah = {}
        langkah_arah = {}
        konvergen_arah = {}

        for a in ["x", "y", "z"]:
            if a in csv_info:
                row = csv_info[a]
                if phi_total is None and "phi" in row:
                    phi_total = float(row["phi"])
                if phi_terhubung is None and "phi_terhubung" in row:
                    phi_terhubung = float(row["phi_terhubung"])
                if "K_mD" in row:
                    k_arah[a] = float(row["K_mD"])
                if "langkah" in row:
                    langkah_arah[a] = int(row["langkah"])
                if "konvergen" in row:
                    konvergen_arah[a] = (int(row["konvergen"]) == 1)

        # Hitung k_geo acuan jika ada
        if len(k_arah) == 3 and all(v > 0 for v in k_arah.values()):
            k_geo = math.exp((math.log(k_arah["x"]) + math.log(k_arah["y"]) + math.log(k_arah["z"])) / 3.0)
        elif len(k_arah) > 0:
            k_geo = float(sum(k_arah.values()) / len(k_arah))
        else:
            k_geo = 100.0

        nama_lengkap = csv_info.get("x", {}).get("nama", nama_default)
        if not nama_lengkap or nama_lengkap == kunci:
            nama_lengkap = nama_default

        katalog[kunci] = {
            "kunci": kunci,
            "nama": nama_lengkap,
            "berkas": nama_file,
            "jalur_berkas": jalur_file,
            "tipe": tipe_default,
            "phi_acuan": phi_total if phi_total is not None else 0.20,
            "phi_th_acuan": phi_terhubung if phi_terhubung is not None else 0.19,
            "K_mD_acuan": k_arah if k_arah else {"x": k_geo, "y": k_geo, "z": k_geo},
            "K_geo_acuan": k_geo,
            "langkah_acuan": langkah_arah,
            "konvergen_acuan": konvergen_arah
        }

    return katalog


def urutkan_daftar_batuan(katalog):
    """
    Mengurutkan seluruh sampel batuan secara ketat berdasarkan porositas acuan (phi_acuan)
    dari yang paling besar ke yang paling kecil.
    """
    return sorted(katalog.keys(), key=lambda k: katalog[k].get("phi_acuan", 0.0), reverse=True)


def dapatkan_batas_langkah_batuan(katalog, batuan, tingkat=0, arah="x"):
    """
    Batas maksimum langkah LBM D3Q19 (simulasi tetap berhenti lebih awal bila sisa < tol).

    - Resolusi penuh T0 (256^3), dari data acuan CSV:
      * Parker (porositas < 10 %, riwayat menunjukkan peluruhan sangat lambat): 2.000.000 langkah.
      * Batuan yang belum konvergen pada acuan: 4 x max(40.000, langkah acuan + 15.000).
      * Batuan lain: 4 x max(15.000, langkah acuan + 5.000).
    - Tingkat diturunkan T1-T3: SAMA dengan batas T0.
      Uji Bentheimer (2026-11-19) menunjukkan tingkat kasar memerlukan cacah langkah yang setara
      dengan T0 untuk mencapai sisa < 5e-5 pada kriteria perubahan nisbi K antar pemeriksaan
      (T2 dan T3 arah z belum konvergen pada 6.000-8.000 langkah). Penskalaan N^2 yang dipakai
      sebelumnya terlalu optimistis. Biaya tambahan kecil karena domain kasar 8-512 kali lebih kecil.
    """
    from program import mode_uji
    if mode_uji.aktif():
        return mode_uji.batas_langkah()
    if batuan == "Parker":
        return 2000000

    info = katalog.get(batuan, {})
    acuan_step = int(info.get("langkah_acuan", {}).get(arah, 15000))
    konv = bool(info.get("konvergen_acuan", {}).get(arah, True))
    # x8: kriteria P7/P11 (sisa < 1e-6 dan fluks antaririsan < 1e-3) memerlukan 2-3x langkah acuan
    # pada ambang tengah, dan lebih banyak lagi pada ambang paling ketat (BSG delta=-0,5 arah z belum
    # tunak pada x4 = 116.000 langkah). Batas ini hanya pengaman: simulasi berhenti sendiri begitu
    # kriteria terpenuhi, sehingga kasus yang sudah tunak tidak terpengaruh.
    return 8 * (max(40000, acuan_step + 15000) if not konv else max(15000, acuan_step + 5000))


def dapatkan_subdirektori_batuan(out_dir, batuan):
    """Mengembalikan path folder penyimpanan mandiri untuk batuan tertentu."""
    sub = os.path.join(out_dir, batuan)
    os.makedirs(sub, exist_ok=True)
    os.makedirs(os.path.join(sub, "grafik"), exist_ok=True)
    os.makedirs(os.path.join(sub, "simulasi_multiresolusi"), exist_ok=True)
    return sub


# Kriteria konvergensi yang berlaku (P7 "Kriteria Konvergensi", P11 sifat_efektif):
# sisa |dK/K| antar pemeriksaan < 1e-6 DAN simpangan nisbi fluks antaririsan < 1e-3, dua pemeriksaan berturut.
KRITERIA_KONVERGENSI = {"tol": 1e-6, "tol_fluks": 1e-3, "periksa": 500, "berturut": 2}


def hasil_sah(d):
    """True bila hasil simulasi tersimpan dibuat dengan kriteria konvergensi yang berlaku,
    atau domainnya tidak tembus (k = 0, tanpa simulasi)."""
    if not d:
        return False
    from program import mode_uji
    if (d.get("mode") == "uji") != mode_uji.aktif():
        return False          # hasil uji dan hasil produksi tidak boleh saling dipakai ulang
    if mode_uji.aktif() and d.get("sisi_uji") != mode_uji.ukuran():
        return False
    if float(d.get("phi_th", 1.0)) == 0.0 and int(d.get("langkah", 0)) == 0:
        return True
    k = d.get("kriteria_konvergensi", {})
    return all(k.get(n) == v for n, v in KRITERIA_KONVERGENSI.items() if n != "periksa")


def tunak_sah(d):
    """Hasil sah DAN tunak. Dalam mode uji, kasus uji yang sudah mencapai batas langkah uji dianggap
    'tunak' agar alur antar-langkah bisa diuji; angka fisisnya tidak untuk laporan."""
    if not hasil_sah(d):
        return False
    if d.get("konvergen") is True:
        return True
    from program import mode_uji
    return mode_uji.aktif() and int(d.get("langkah", 0)) >= mode_uji.batas_langkah()
