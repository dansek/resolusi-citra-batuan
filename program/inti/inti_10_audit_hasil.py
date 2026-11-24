#!/usr/bin/env python3
"""
program/inti/inti_10_audit_hasil.py
===================================
Audit keluaran pipeline: sumber data, kelengkapan metadata, dan hitung ulang mandiri semua angka turunan.
Dijalankan setelah setiap tahap (mode uji maupun produksi) dan wajib bersih sebelum hasil dipakai di laporan.

  python program/inti/inti_10_audit_hasil.py --backend gpu --rocks Bentheimer BSG
  python program/inti/inti_10_audit_hasil.py --out_dir hasil_uji --backend gpu --banding cpu --rocks Bentheimer BSG

A  Berkas kasus LBM: backend sesuai folder, mode (uji/produksi) sesuai folder, kriteria berlaku, N sesuai
   tingkat, metadata lengkap, tidak ada K yang sama persis dengan data acuan dosen, daftar kasus belum tunak.
A2 Jangkar data acuan dosen (produksi, 256^3): phi dan phi_th T0 harus SAMA PERSIS (geometri murni,
   tanpa LBM); K T0 per arah harus dalam 1% (pemecah dan kriteria berhenti berbeda dari acuan).
B  Rantai data: T0 langkah 05 = langkah 01; k_ref langkah 02 = langkah 01; tabel dua cara = berkas kasus.
C  Hitung ulang: Delta_seg, simpangan k_geo, tingkat kritis (berhenti di kegagalan pertama), lipatan FOV,
   galat_k P2, pita tiga sifat dan tingkat runtuh.
D  (--banding) paritas antar-backend: selisih nisbi terbesar per tabel CSV dan per berkas kasus.
"""

import os
import sys
import csv
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from program.katalog_batuan import KRITERIA_KONVERGENSI

TEMUAN = []
CATATAN = []
DX0 = 2.25
KODE = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}


def masalah(rock, teks):
    TEMUAN.append(f"{rock}: {teks}")


def baca_csv(j):
    with open(j, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baca_json(j):
    with open(j, encoding="utf-8") as f:
        return json.load(f)


def geo(v):
    v = [float(x) for x in v]
    return float(np.exp(np.mean(np.log(v)))) if all(x > 0 for x in v) else 0.0


def dekat(a, b, rtol=1e-9, atol=1e-9):
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def bagian_a(rock, dirb, backend, uji, acuan_k):
    berkas = sorted(glob.glob(os.path.join(dirb, "**", "*.json"), recursive=True))
    n, belum = 0, []
    for j in berkas:
        try:
            d = baca_json(j)
        except Exception as e:
            masalah(rock, f"JSON rusak {os.path.relpath(j, dirb)}: {e}")
            continue
        if not isinstance(d, dict) or "langkah" not in d or "K_mD" not in d or isinstance(d.get("K_mD"), dict):
            continue
        n += 1
        nama = os.path.relpath(j, dirb)
        if d.get("backend") != backend:
            masalah(rock, f"{nama}: backend {d.get('backend')} di folder {backend}")
        if (d.get("mode") == "uji") != uji:
            masalah(rock, f"{nama}: mode {d.get('mode', 'produksi')} tidak sesuai folder")
        tembus = not (float(d.get("phi_th", 1)) == 0.0 and int(d.get("langkah", 0)) == 0)
        if tembus:
            k = d.get("kriteria_konvergensi") or {}
            if any(k.get(a) != b for a, b in KRITERIA_KONVERGENSI.items() if a != "periksa"):
                masalah(rock, f"{nama}: kriteria {k} bukan kriteria berlaku")
            for f in ("riwayat", "diagnostik", "n_fluida"):
                if not d.get(f):
                    masalah(rock, f"{nama}: field '{f}' kosong")
            if not d.get("memori"):
                CATATAN.append(f"{rock}: {nama} tanpa field memori (dibuat sebelum pencatatan memori)")
            if d.get("konvergen") is not True:
                belum.append((nama, d.get("langkah")))
        for a, kref in acuan_k.items():
            if float(d["K_mD"]) > 0 and dekat(d["K_mD"], kref, rtol=1e-12, atol=0):
                masalah(rock, f"{nama}: K = K acuan dosen arah {a} (salinan?)")
        t = d.get("tingkat")
        if "simulasi_multiresolusi" in j and t is not None:
            n0 = d.get("sisi_uji") or 256
            if int(d.get("N", -1)) != n0 // 2 ** int(t):
                masalah(rock, f"{nama}: N={d.get('N')} padahal tingkat {t} -> {n0 // 2 ** int(t)}")
    return n, belum


def bagian_a2(rock, dirb, acuan_penuh):
    p01 = baca_json(os.path.join(dirb, "permeabilitas_kondisi_tunak.json"))
    baris = []
    for a, r in acuan_penuh.items():
        if not dekat(p01["phi_total"], r["phi_abs"], rtol=1e-12):
            masalah(rock, f"phi T0 {p01['phi_total']} != acuan {r['phi_abs']}")
        if not dekat(p01["phi_tersambung"], r["phi_terhubung"], rtol=1e-12):
            masalah(rock, f"phi_th T0 {p01['phi_tersambung']} != acuan {r['phi_terhubung']} (arah {a})")
        e = p01["K_mD"][a] / r["K_mD"] - 1
        baris.append(f"{a} {100*e:+.2f}%")
        if abs(e) > 0.01:
            masalah(rock, f"K T0 arah {a} menyimpang {100*e:+.2f}% dari acuan dosen (> 1%)")
    return baris


def bagian_b(rock, dirb):
    p01 = baca_json(os.path.join(dirb, "permeabilitas_kondisi_tunak.json"))
    for a in "xyz":
        m = baca_json(os.path.join(dirb, "simulasi_multiresolusi", f"{rock}_T0_resolusi_penuh_{a}.json"))
        if not dekat(m["K_mD"], p01["K_mD"][a]):
            masalah(rock, f"T0 {a}: langkah 05 {m['K_mD']} != langkah 01 {p01['K_mD'][a]}")
        r = p01["riwayat_konvergensi"][a]
        if r.get("sumber") != "simulasi backend ini":
            masalah(rock, f"langkah 01 arah {a}: sumber = {r.get('sumber')}")
    s02 = baca_json(os.path.join(dirb, "rentang_segmentasi_acuan.json"))
    for a in "xyz":
        if not dekat(s02[f"k{a}_ref_mD"], p01["K_mD"][a]):
            masalah(rock, f"k_ref langkah 02 arah {a} != langkah 01")
    rows = baca_csv(os.path.join(dirb, "tabel_dua_cara_penurunan.csv"))
    for r in rows:
        if r["tingkat"] == "T0":
            continue
        ks = []
        for a in "xyz":
            f = glob.glob(os.path.join(dirb, "simulasi_multiresolusi", f"{rock}_{r['tingkat']}_{r['cara']}_{a}.json"))
            if len(f) != 1:
                masalah(rock, f"berkas kasus {r['tingkat']} {r['cara']} {a} tidak tunggal: {f}")
                break
            ks.append(baca_json(f[0])["K_mD"])
        else:
            if not dekat(geo(ks), r["k_geo_mD"], rtol=1e-9):
                masalah(rock, f"tabel dua cara {r['tingkat']} {r['cara']}: k_geo {r['k_geo_mD']} != {geo(ks)}")
    return p01, s02, rows


def bagian_c(rock, dirb, s02):
    k_ref = float(s02["k_geo_ref_mD"])
    ren = baca_csv(os.path.join(dirb, "tabel_rentang_segmentasi.csv"))
    kg = [float(r["k_mD"]) for r in ren]
    dseg = (max(kg) - min(kg)) / (2 * float(np.median(kg))) * 100        # P16 Soal 5: terhadap median sapuan
    for r in ren:
        if not dekat(r["k_mD"], geo([r["kx_mD"], r["ky_mD"], r["kz_mD"]]), rtol=1e-9):
            masalah(rock, f"rentang indeks {r['indeks']}: k_mD bukan rerata geometrik kx,ky,kz")
    sim = baca_csv(os.path.join(dirb, "tabel_simpangan_permeabilitas.csv"))
    dua = {(r["tingkat"], r["cara"]): r for r in baca_csv(os.path.join(dirb, "tabel_dua_cara_penurunan.csv"))}
    for r in sim:
        harap = (float(dua[(r["tingkat"], r["cara"])]["k_geo_mD"]) / k_ref - 1) * 100
        if not dekat(r["delta_kgeo_persen"], harap, rtol=1e-9, atol=1e-9):
            masalah(rock, f"simpangan {r['tingkat']} {r['cara']}: {r['delta_kgeo_persen']} != {harap}")
        if (str(abs(harap) <= dseg) != r["dalam_pita_geo"]):
            masalah(rock, f"dalam_pita_geo {r['tingkat']} {r['cara']} salah")
    for r in baca_csv(os.path.join(dirb, "tabel_tingkat_kritis_fov.csv")):
        if not dekat(r["setengah_lebar_segmentasi_persen"], dseg, rtol=1e-9):
            masalah(rock, f"Delta_seg tercatat {r['setengah_lebar_segmentasi_persen']} != hitung ulang {dseg}")
        kritis = "T0"
        for t in ("T1", "T2", "T3"):
            d = (float(dua[(t, r["cara"])]["k_geo_mD"]) / k_ref - 1) * 100
            if abs(d) <= dseg:
                kritis = t
            else:
                break
        if r["tingkat_kritis"] != kritis:
            masalah(rock, f"tingkat kritis {r['cara']}: tercatat {r['tingkat_kritis']}, hitung ulang {kritis}")
        if "tingkat_kritis_nyquist" in r:
            d_t = float(baca_json(os.path.join(dirb, "karakteristik_leher_pori.json")).get("dt_um")
                        or baca_json(os.path.join(dirb, "karakteristik_leher_pori.json"))[rock]["dt_um"])
            kn = "T0"
            for t in ("T1", "T2", "T3"):
                dxt = DX0 * 2 ** int(t[1])
                d = (float(dua[(t, r["cara"])]["k_geo_mD"]) / k_ref - 1) * 100
                if abs(d) <= dseg and d_t / dxt >= 2.0:
                    kn = t
                else:
                    break
            if r["tingkat_kritis_nyquist"] != kn:
                masalah(rock, f"tingkat kritis Nyquist {r['cara']}: tercatat {r['tingkat_kritis_nyquist']}, hitung ulang {kn}")
        dxk = DX0 * 2 ** int(kritis[1])
        if not (dekat(r["dx_kritis_um"], dxk) and dekat(r["lipatan_fov_per_arah"], dxk / DX0)
                and dekat(r["lipatan_fov_volume"], (dxk / DX0) ** 3)):
            masalah(rock, f"dx kritis / lipatan FOV {r['cara']} salah")
    for r in baca_csv(os.path.join(dirb, "tabel_evaluasi_kelayakan_resolusi.csv")):
        nis = float(r["dt_per_dx"])
        if not dekat(r["galat_k_persen"], 200.0 / nis, rtol=1e-9):
            masalah(rock, f"galat_k {r['tingkat']}: {r['galat_k_persen']} != 2/(d_t/dx)")
        if (nis < 2) != (r["di_bawah_nyquist"] == "True"):
            masalah(rock, f"bendera Nyquist {r['tingkat']} salah")
    # tiga sifat: pita = (maks - min) / (2 nilai T0) dari tabel rentang
    tiga = baca_csv(os.path.join(dirb, "tabel_komparasi_tiga_sifat.csv"))
    t0 = next(r for r in tiga if r["tingkat"] == "T0")
    for sifat, kol_ren, kol_t in (("phi", "phi_tot", "phi_tot"), ("Sv", "Sv_um1", "Sv_um1"), ("k", "k_mD", "k_geo_mD")):
        v = [float(r[kol_ren]) for r in ren]
        ref = float(t0[kol_t]) if sifat != "k" else k_ref
        pita = (max(v) - min(v)) / (2 * float(np.median(v))) * 100
        for r in tiga:
            simp = (float(r[kol_t]) / ref - 1) * 100
            if not dekat(r[f"simpangan_{sifat}_persen"], simp, rtol=1e-9, atol=1e-9):
                masalah(rock, f"tiga sifat {sifat} {r['tingkat']} {r['cara']}: simpangan {r[f'simpangan_{sifat}_persen']} != {simp}")
            if str(abs(simp) <= pita) != r[f"dalam_pita_{sifat}"]:
                masalah(rock, f"tiga sifat {sifat} {r['tingkat']} {r['cara']}: bendera pita salah (pita {pita:.3f}%)")
    return dseg


def bagian_d(rock, dir_a, dir_b):
    hasil = []
    for j in sorted(glob.glob(os.path.join(dir_a, "*.csv"))):
        jb = os.path.join(dir_b, os.path.basename(j))
        if not os.path.exists(jb):
            masalah(rock, f"paritas: {os.path.basename(j)} tidak ada di pembanding")
            continue
        ra, rb = baca_csv(j), baca_csv(jb)
        terbesar, kolom = 0.0, ""
        for x, y in zip(ra, rb):
            for k in x:
                try:
                    a, b = float(x[k]), float(y[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(a) and np.isfinite(b) and max(abs(a), abs(b)) > 0:
                    e = abs(a - b) / max(abs(a), abs(b))
                    if e > terbesar:
                        terbesar, kolom = e, k
        hasil.append((os.path.basename(j), terbesar, kolom))
    kasus = 0
    terbesar = 0.0
    for j in glob.glob(os.path.join(dir_a, "**", "*.json"), recursive=True):
        jb = j.replace(dir_a, dir_b)
        if not os.path.exists(jb):
            continue
        a, b = baca_json(j), baca_json(jb)
        if isinstance(a, dict) and "langkah" in a and isinstance(a.get("K_mD"), float) and isinstance(b.get("K_mD"), float):
            kasus += 1
            if max(a["K_mD"], b["K_mD"]) > 0:
                terbesar = max(terbesar, abs(a["K_mD"] - b["K_mD"]) / max(a["K_mD"], b["K_mD"]))
            if a.get("langkah") != b.get("langkah"):
                masalah(rock, f"paritas: langkah berbeda {os.path.relpath(j, dir_a)} {a['langkah']} vs {b['langkah']}")
    return hasil, kasus, terbesar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "hasil"))
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--banding", default=None, choices=["gpu", "cpu"], help="backend pembanding (paritas)")
    ap.add_argument("--rocks", nargs="+", default=["Bentheimer", "BSG"])
    ap.add_argument("--toleransi_paritas", type=float, default=1e-6)
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)
    uji = os.path.basename(out_dir.rstrip("/")) == "hasil_uji"
    print(f"AUDIT {out_dir}/{a.backend} ({'mode uji' if uji else 'produksi'}) : {', '.join(a.rocks)}")
    for rock in a.rocks:
        dirb = os.path.join(out_dir, a.backend, rock)
        acuan, acuan_penuh = {}, {}
        for ax in "xyz":
            j = os.path.join(ROOT, "data", "sumber", "riwayat-konvergensi", f"{rock}_penuh_{ax}.json")
            if os.path.exists(j):
                acuan_penuh[ax] = baca_json(j)
                acuan[ax] = float(acuan_penuh[ax]["K_mD"])
        print(f"\n== {rock}")
        n, belum = bagian_a(rock, dirb, a.backend, uji, acuan)
        print(f"  A  {n} berkas kasus; belum tunak: {len(belum)}" + ("" if uji else "".join(f"\n       - {x} ({y} langkah)" for x, y in belum)))
        if belum and not uji:
            masalah(rock, f"{len(belum)} kasus belum tunak")
        if not uji and acuan_penuh:
            try:
                print(f"  A2 jangkar acuan dosen: phi & phi_th diperiksa persis; K T0 vs acuan {', '.join(bagian_a2(rock, dirb, acuan_penuh))}")
            except FileNotFoundError as e:
                masalah(rock, f"berkas hilang: {e.filename}")
        try:
            _, s02, _ = bagian_b(rock, dirb)
            print("  B  rantai data diperiksa")
            dseg = bagian_c(rock, dirb, s02)
            print(f"  C  hitung ulang selesai (Delta_seg = {dseg:.2f}%)")
        except FileNotFoundError as e:
            masalah(rock, f"berkas hilang: {e.filename}")
        if a.banding:
            hasil, kasus, kmax = bagian_d(rock, dirb, os.path.join(out_dir, a.banding, rock))
            print(f"  D  paritas {a.backend} vs {a.banding}: {kasus} berkas kasus, selisih K terbesar {kmax:.2e}")
            for f, e, k in hasil:
                tanda = "" if e <= a.toleransi_paritas else "   <-- melebihi toleransi"
                print(f"       {f:<42} {e:.2e} {k}{tanda}")
                if e > a.toleransi_paritas:
                    masalah(rock, f"paritas {f}: {e:.2e} pada kolom {k}")
            if kmax > a.toleransi_paritas:
                masalah(rock, f"paritas K berkas kasus {kmax:.2e}")
    if CATATAN:
        print(f"\nCatatan ({len(CATATAN)}):")
        for c in sorted(set(CATATAN))[:10]:
            print("  -", c)
        if len(set(CATATAN)) > 10:
            print(f"  ... dan {len(set(CATATAN)) - 10} lagi")
    print("\n" + ("AUDIT BERSIH" if not TEMUAN else f"{len(TEMUAN)} TEMUAN:"))
    for t in TEMUAN:
        print("  -", t)
    sys.exit(1 if TEMUAN else 0)


if __name__ == "__main__":
    main()
