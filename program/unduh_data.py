#!/usr/bin/env python3
"""
program/unduh_data.py
=====================
Mengunduh citra biner 256^3 dari folder Google Drive publik ke data/sumber/citra-biner/.

Daftar berkas, ukuran, dan sidik SHA-256 diambil dari data/metadata.json (bagian "berkas" dan "unduhan").
Berkas yang sudah ada dan sidiknya cocok dilewati; berkas rusak atau tidak lengkap diunduh ulang.
Berkas yang diunduh baru dipakai setelah sidiknya cocok, sehingga simulasi tidak pernah berjalan
pada citra yang berbeda dengan citra yang dipakai di laporan.

ID berkas diambil dari "id_drive" di metadata; bila kosong, ID dicari dari daftar isi folder publik.
Hanya pustaka standar Python yang dipakai.

  python program/unduh_data.py                    # Bentheimer dan BSG (batuan laporan)
  python program/unduh_data.py --rocks all        # seluruh 11 batuan
  python program/unduh_data.py --rocks Berea BUG
  python program/unduh_data.py --periksa          # hanya periksa berkas lokal, tanpa mengunduh
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA = os.path.join(ROOT, "data", "metadata.json")
BATUAN_LAPORAN = ["Bentheimer", "BSG"]
AGEN = {"User-Agent": "Mozilla/5.0 (unduh-citra-s4)"}
COBA = 3


def muat_metadata():
    with open(METADATA, encoding="utf-8") as f:
        return json.load(f)


def daftar_berkas(meta):
    return {b["batuan"]: b for b in meta["berkas"]}


def tujuan_lokal(meta, data_dir=None):
    if data_dir:
        return os.path.join(os.path.abspath(data_dir), "sumber", "citra-biner")
    return os.path.join(ROOT, meta["unduhan"]["tujuan_lokal"])


def sidik(jalur):
    h = hashlib.sha256()
    with open(jalur, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def sah(jalur, info):
    return os.path.exists(jalur) and os.path.getsize(jalur) == info["bait"] and sidik(jalur) == info["sha256"]


def _ambil(url, batas_waktu=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGEN), timeout=batas_waktu) as r:
        return r.read()


def isi_folder(id_folder):
    """Daftar {nama: id} dari tampilan tersemat folder Drive publik."""
    teks = _ambil(f"https://drive.google.com/embeddedfolderview?id={id_folder}#list").decode("utf-8", "replace")
    isi = {}
    for m in re.finditer(r'id="entry-([A-Za-z0-9_-]+)".*?class="flip-entry-title">([^<]*)<', teks, re.S):
        isi[html.unescape(m.group(2)).strip()] = m.group(1)
    return isi


def cari_id(meta, nama_berkas, _cache={}):
    u = meta["unduhan"]
    if "isi" not in _cache:
        id_induk = re.search(r"folders/([A-Za-z0-9_-]+)", u["folder_drive"]).group(1)
        induk = isi_folder(id_induk)
        sub = u.get("subfolder")
        _cache["isi"] = isi_folder(induk[sub]) if sub and sub in induk else induk
    return _cache["isi"].get(nama_berkas)


def unduh_satu(info, jalur, meta):
    id_drive = info.get("id_drive") or cari_id(meta, info["berkas"])
    if not id_drive:
        raise RuntimeError(f"{info['berkas']} tidak ditemukan di folder Drive {meta['unduhan']['folder_drive']}")
    url = f"https://drive.usercontent.google.com/download?id={id_drive}&export=download&confirm=t"
    sementara = jalur + ".sebagian"
    for k in range(1, COBA + 1):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGEN), timeout=120) as r, open(sementara, "wb") as f:
                for blok in iter(lambda: r.read(1 << 20), b""):
                    f.write(blok)
                    h.update(blok)
            if os.path.getsize(sementara) == info["bait"] and h.hexdigest() == info["sha256"]:
                os.replace(sementara, jalur)
                return
            print(f"    [!] {info['berkas']}: sidik tidak cocok (percobaan {k}/{COBA})", flush=True)
        except OSError as e:
            print(f"    [!] {info['berkas']}: {e} (percobaan {k}/{COBA})", flush=True)
        time.sleep(2 * k)
    if os.path.exists(sementara):
        os.remove(sementara)
    raise RuntimeError(f"{info['berkas']} gagal diunduh dengan sidik yang benar setelah {COBA} percobaan")


def pastikan_citra(rocks=None, data_dir=None, unduh=True):
    """Memastikan citra batuan tersedia dan sidiknya cocok. Mengembalikan daftar batuan yang belum siap."""
    meta = muat_metadata()
    semua = daftar_berkas(meta)
    if rocks is None:
        rocks = BATUAN_LAPORAN
    elif len(rocks) == 1 and rocks[0].lower() in ("all", "semua"):
        rocks = sorted(semua, key=str.lower)
    tak_dikenal = [r for r in rocks if r not in semua]
    if tak_dikenal:
        raise SystemExit(f"[GALAT] Batuan tidak dikenal: {', '.join(tak_dikenal)}. Pilihan: {', '.join(sorted(semua, key=str.lower))}")
    folder = tujuan_lokal(meta, data_dir)
    os.makedirs(folder, exist_ok=True)
    belum = []
    for r in rocks:
        info = semua[r]
        jalur = os.path.join(folder, info["berkas"])
        if sah(jalur, info):
            print(f"  [ada]   {info['berkas']} (sidik cocok)", flush=True)
            continue
        if os.path.exists(jalur):
            print(f"  [rusak] {info['berkas']} berbeda dari sidik metadata", flush=True)
        if not unduh:
            belum.append(r)
            continue
        print(f"  [unduh] {info['berkas']} ({info['bait'] / 2**20:.0f} MB) ...", flush=True)
        try:
            unduh_satu(info, jalur, meta)
            print(f"  [ok]    {info['berkas']} (sidik cocok)", flush=True)
        except (RuntimeError, OSError) as e:
            print(f"  [GAGAL] {e}", flush=True)
            belum.append(r)
    return belum


def main():
    ap = argparse.ArgumentParser(description="Unduh citra biner batupasir dari Google Drive")
    ap.add_argument("--rocks", nargs="+", default=None, help="nama batuan atau 'all' (bawaan: Bentheimer BSG)")
    ap.add_argument("--data_dir", default=None, help="folder data (bawaan: data/)")
    ap.add_argument("--periksa", action="store_true", help="hanya periksa berkas lokal, tanpa mengunduh")
    args = ap.parse_args()
    belum = pastikan_citra(args.rocks, args.data_dir, unduh=not args.periksa)
    if belum:
        sys.exit(f"[GALAT] Citra belum siap: {', '.join(belum)}")
    print("Semua citra siap.")


if __name__ == "__main__":
    main()
