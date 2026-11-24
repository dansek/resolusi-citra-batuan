#!/usr/bin/env python3
"""
program/angka_laporan.py
========================
Menghitung angka turunan yang dikutip laporan dari hasil tersimpan, tanpa simulasi, lalu menuliskannya ke
hasil/angka_laporan.csv beserta berkas asal tiap angka.

  python program/angka_laporan.py            # hasil GPU sebagai utama, hasil CPU sebagai pembanding
  python program/angka_laporan.py --backend cpu
"""
import argparse
import csv
import glob
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HASIL = os.path.join(ROOT, "hasil")
DATA = os.path.join(ROOT, "data")
BATUAN = ("Bentheimer", "BSG")
ARAH = "xyz"
DARCY_M2 = 9.869233e-13
BARIS = []


def baca(jalur):
    with open(jalur, encoding="utf-8") as f:
        return json.load(f)


def catat(bagian, nama, batuan, nilai, satuan, sumber):
    BARIS.append({"bagian": bagian, "besaran": nama, "batuan": batuan,
                  "nilai": f"{nilai:.6g}" if isinstance(nilai, float) else nilai, "satuan": satuan, "sumber": sumber})


def geo(nilai):
    return math.exp(sum(math.log(v) for v in nilai) / len(nilai))


def kasus(backend_dir, b):
    keluaran = {}
    for f in glob.glob(os.path.join(backend_dir, b, "**", f"{b}_*.json"), recursive=True):
        d = baca(f)
        if d.get("langkah") and d.get("riwayat") and len(d["riwayat"][0]) >= 6:
            keluaran[os.path.relpath(f, os.path.join(backend_dir, b))] = d
    return keluaran


def indeks_berturut(r, syarat, n):
    s = 0
    for i, v in enumerate(r):
        s = s + 1 if syarat(v) else 0
        if s >= n:
            return i
    return None


def main(hasil=HASIL, utama="gpu"):
    """utama = backend yang angkanya dikutip; backend lain hanya dipakai untuk paritas GPU-CPU."""
    lain = "cpu" if utama == "gpu" else "gpu"
    U, L = os.path.join(hasil, utama), os.path.join(hasil, lain)
    acuan = {}
    with open(os.path.join(DATA, "sumber", "sebelas-batupasir.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acuan[(r["batuan"], r["arah"])] = r
    with open(os.path.join(DATA, "pembanding_eksternal.csv"), encoding="utf-8") as f:
        luar = {r["batuan"]: r for r in csv.DictReader(f)}
    jalur_waktu = os.path.join(hasil, "waktu_maraton.csv")
    if not os.path.exists(jalur_waktu):          # run ulang ke folder baru: waktu produksi laporan tetap dari hasil/
        jalur_waktu = os.path.join(HASIL, "waktu_maraton.csv")
    with open(jalur_waktu, encoding="utf-8") as f:
        waktu = {(r["backend"], r["batuan"]): int(r["menit"]) for r in csv.DictReader(f)}

    semua_rasio, k_geo, koponen, langkah_t0 = [], {}, {}, {}
    selisih_gpu_cpu, n_pasangan = 0.0, 0
    for b in BATUAN:
        g, c = kasus(U, b), kasus(L, b)
        for nama, d in g.items():
            if nama in c:
                n_pasangan += 1
                selisih_gpu_cpu = max(selisih_gpu_cpu, abs(d["K_mD"] - c[nama]["K_mD"]) / d["K_mD"])
        beban = [d["kondisi_perangkat"]["beban_cpu_lain_selama_run"] for x in (g, c) for d in x.values()
                 if d.get("kondisi_perangkat")]
        catat("2.3", "beban proses lain terbesar", b, 100 * max(beban), "%", "kondisi_perangkat pada berkas kasus")
        catat("2.3", "waktu rantai kerja GPU", b, waktu[("gpu", b)], "menit", "hasil/waktu_maraton.csv")
        catat("2.3", "waktu rantai kerja CPU", b, waktu[("cpu", b)], "menit", "hasil/waktu_maraton.csv")
        catat("2.3", "percepatan GPU", b, waktu[("cpu", b)] / waktu[("gpu", b)], "kali", "hasil/waktu_maraton.csv")

        fluks_akhir, geser_t, geser_semua = 0, 0.0, 0.0
        for nama, d in g.items():
            r = d["riwayat"]
            i_s = indeks_berturut(r, lambda v: v[2] < 1e-6, 2)
            i_f = indeks_berturut(r, lambda v: v[5] < 1e-3, 2)
            fluks_akhir += int(i_s is not None and i_f is not None and i_f > i_s)
            i_a = indeks_berturut(r, lambda v: v[2] < 5e-5, 3)
            i_a = len(r) - 1 if i_a is None else i_a
            geser = abs(r[i_a][1] / r[-1][1] - 1) * 100
            geser_semua = max(geser_semua, geser)
            if nama.startswith("simulasi_multiresolusi"):
                geser_t = max(geser_t, geser)
        catat("3", "kasus dengan syarat fluks terpenuhi sesudah sisa", b, fluks_akhir, f"dari {len(g)}", f"riwayat kasus {utama.upper()}")
        catat("3", "geser k maks bila berhenti dengan kriteria acuan, T0-T3", b, geser_t, "%", f"riwayat kasus {utama.upper()}")
        catat("3", "geser k maks bila berhenti dengan kriteria acuan, semua kasus", b, geser_semua, "%", f"riwayat kasus {utama.upper()}")

        ks, taus, langkah_t0[b] = [], [], {}
        rek = baca(os.path.join(U, b, "rekapitulasi_akhir_s4.json"))
        t0 = next(x for x in rek["tabel_sifat_efektif"] if x["tingkat"] == "T0")
        sifat = next(x for x in rek["tabel_tiga_sifat"] if x["tingkat"] == "T0")
        phi, sv = sifat["phi_tot"], sifat["Sv_um1"]
        kop = []
        for a in ARAH:
            d = baca(os.path.join(U, b, "simulasi_multiresolusi", f"{b}_T0_resolusi_penuh_{a}.json"))
            r, ac = d["riwayat"], acuan[(b, a)]
            la, ka = int(ac["langkah"]), float(ac["K_mD"])
            faktor = d["K_mD"] / r[-1][1]
            k_saat_acuan = next(v[1] for v in r if v[0] == la) * faktor
            i_f = indeks_berturut(r, lambda v: v[5] < 1e-3, 2)
            langkah_t0[b][a] = d["langkah"]
            semua_rasio.append(d["langkah"] / la)
            catat("3", f"langkah fluks < 1e-3, arah {a}", b, r[i_f][0], "langkah", "riwayat T0")
            catat("3", f"selisih k terhadap acuan pada langkah acuan, arah {a}", b, (k_saat_acuan / ka - 1) * 100, "%", "riwayat T0")
            catat("5", f"selisih k akhir terhadap acuan, arah {a}", b, (d["K_mD"] / ka - 1) * 100, "%", "riwayat T0")
            catat("5", f"langkah tunak, arah {a}", b, d["langkah"], "langkah", "riwayat T0")
            tau = t0[f"tau_u_{a}"]
            kop.append(phi ** 3 / (2 * tau ** 2 * sv ** 2) * 1e-12 / DARCY_M2 * 1e3)
            ks.append(d["K_mD"])
        k_geo[b], koponen[b] = geo(ks), geo(kop)
        lab = float(luar[b]["k_laboratorium_mD"])
        catat("5", "k Koponen", b, koponen[b], "mD", "phi, tau_u, Sv hasil citra ini")
        catat("5", "k simulasi / k Koponen", b, k_geo[b] / koponen[b], "kali", "dihitung")
        catat("5", "k simulasi", b, k_geo[b], "mD", "simulasi_multiresolusi T0")
        catat("5", "k laboratorium", b, lab, "mD", "data/pembanding_eksternal.csv")
        catat("5", "k simulasi / k laboratorium", b, k_geo[b] / lab, "kali", "dihitung")
        catat("5", "selisih phi citra terhadap rerata formasi", b, (phi / float(luar[b]["phi_rerata_formasi"]) - 1) * 100, "%", "data/pembanding_eksternal.csv")

        seg = baca(os.path.join(U, b, "basis_acuan.json"))
        med = seg["k_median_sapuan_mD"]
        catat("6", "k maks terhadap median", b, (seg["k_max_mD"] / med - 1) * 100, "%", "basis_acuan.json")
        catat("6", "k min terhadap median", b, (seg["k_min_mD"] / med - 1) * 100, "%", "basis_acuan.json")
        catat("6", "k median terhadap citra asli", b, (med / seg["k_geo_ref_mD"] - 1) * 100, "%", "basis_acuan.json")
        terlama = max((d for n, d in g.items() if n.startswith("rentang_segmentasi")), key=lambda d: d["langkah"])
        catat("6", "langkah terbanyak pada sapuan segmentasi", b, terlama["langkah"], "langkah", "rentang_segmentasi/")

        catat("2.2", "k_geo data acuan", b, geo([float(acuan[(b, a)]["K_mD"]) for a in ARAH]), "mD", "data/sumber/sebelas-batupasir.csv")
        tunak = baca(os.path.join(U, b, "permeabilitas_kondisi_tunak.json"))
        catat("2.1", "selisih phi citra terhadap tabel acuan", b, abs(tunak["phi_total"] - float(acuan[(b, "x")]["phi"])), "mutlak",
              "permeabilitas_kondisi_tunak.json dan data/sumber/sebelas-batupasir.csv")
        ang = rek["anggaran_ketidakpastian"]
        for sisi, tanda in (("bawah", -1), ("atas", 1)):
            catat("12", f"selang 95 % k resolusi penuh, batas {sisi}", b, ang["k0_geo_mD"] * math.exp(tanda * ang["U_lnk"]), "mD",
                  "rekapitulasi_akhir_s4.json (k0 e^(+-U))")

        rt = float(luar[b]["r_t_rerata_um"])
        for dx in (4.5, 9.0):
            catat("10", f"2 r_t / dx pada {dx:g} um", b, 2 * rt / dx, "voxel", "data/pembanding_eksternal.csv")

    if n_pasangan:
        catat("2.3", "selisih k GPU-CPU terbesar", "keduanya", selisih_gpu_cpu, f"nisbi, {n_pasangan} pasangan", "hasil/gpu dan hasil/cpu")
    catat("3", "kelipatan langkah terhadap acuan, minimum", "keduanya", min(semua_rasio), "kali", "riwayat T0")
    catat("3", "kelipatan langkah terhadap acuan, maksimum", "keduanya", max(semua_rasio), "kali", "riwayat T0")
    rasio = [langkah_t0["BSG"][a] / langkah_t0["Bentheimer"][a] for a in ARAH]
    catat("5", "kelipatan langkah BSG terhadap Bentheimer, minimum", "keduanya", min(rasio), "kali", "riwayat T0")
    catat("5", "kelipatan langkah BSG terhadap Bentheimer, maksimum", "keduanya", max(rasio), "kali", "riwayat T0")
    catat("5", "nisbah k Bentheimer/BSG, simulasi", "keduanya", k_geo["Bentheimer"] / k_geo["BSG"], "kali", "dihitung")
    catat("5", "nisbah k Bentheimer/BSG, laboratorium", "keduanya",
          float(luar["Bentheimer"]["k_laboratorium_mD"]) / float(luar["BSG"]["k_laboratorium_mD"]), "kali", "data/pembanding_eksternal.csv")
    catat("5", "nisbah k Bentheimer/BSG, Koponen", "keduanya", koponen["Bentheimer"] / koponen["BSG"], "kali", "dihitung")

    keluaran = os.path.join(hasil, "angka_laporan.csv")
    with open(keluaran, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(BARIS[0]))
        w.writeheader()
        w.writerows(BARIS)
    print(f"{len(BARIS)} angka ditulis ke {os.path.relpath(keluaran, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Angka turunan laporan dari hasil tersimpan")
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--hasil", default=HASIL)
    a = ap.parse_args()
    main(os.path.abspath(a.hasil), a.backend)
