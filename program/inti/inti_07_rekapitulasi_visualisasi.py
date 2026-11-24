#!/usr/bin/env python3
"""
program/inti/inti_07_rekapitulasi_visualisasi.py
===========================================
Langkah 07 (Bab 8 langkah 9; luaran wajib 4 dan 6, luaran khusus 13): rekapitulasi akhir.

1. Tabel tiga sifat (luaran 13). Porositas, luas spesifik, dan permeabilitas pada empat tingkat dan dua cara.
   Tiap sifat dibandingkan terhadap PITA SEGMENTASINYA SENDIRI dari sapuan Langkah 02
   (setengah lebar = (maks - min) / (2 nilai citra asli)), sehingga "runtuh" berarti simpangan
   resolusi melampaui ketidakpastian yang sudah ada (Bab 8; P2: kelayakan dinyatakan terhadap sifat
   yang dicari). Tingkat runtuh = tingkat pertama, ditelusuri T1 -> T3, yang keluar dari pitanya.
2. Tabel sifat efektif (luaran 6). k tiga arah dan rerata geometriknya (P11: satu angka pengganti),
   tortuositas aliran tau_u = sum|u| / sum u_aliran pada simpul pori sampel (P11 sifat_aliran).
   Perjanjian perata-rataan: k dari kecepatan Darcy = rerata u_aliran atas SELURUH volume sampel
   (pori dan padatan), bukan atas pori saja (P11: rerata pori memberi k/phi, keliru).
3. Grafik riwayat sisa konvergensi terhadap langkah waktu (luaran 4).
4. Grafik biaya komputasi: durasi dan MLUPS tiap tingkat terhadap simpangan k.
"""

import os
import sys
import csv
import json

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.katalog_batuan import dapatkan_subdirektori_batuan

ARAH = ["x", "y", "z"]
TINGKAT = [(0, "T0", 2.25, ["resolusi_penuh"]), (1, "T1", 4.5, ["majority", "block_avg"]),
           (2, "T2", 9.0, ["majority", "block_avg"]), (3, "T3", 18.0, ["majority", "block_avg"])]
NAMA_CARA = {"resolusi_penuh": "Resolusi penuh", "majority": "Suara terbanyak", "block_avg": "Rerata blok"}
TOL = 1e-6
TOL_FLUKS = 1e-3


def baca_json(j):
    with open(j, encoding="utf-8") as f:
        return json.load(f)


def wajib(j, langkah):
    if not os.path.exists(j):
        raise FileNotFoundError(f"{os.path.basename(j)} tidak ada; jalankan Langkah {langkah} lebih dahulu")
    return j


def geo(v):
    return float(np.exp(np.mean(np.log(v)))) if all(x > 0 for x in v) else 0.0


def setengah_lebar(nilai, ref=None):
    """P16 Soal 5: (maks - min) / 2 / median sapuan x 100 %. `ref` diabaikan (dipertahankan untuk kompatibilitas)."""
    med = float(np.median(nilai))
    return (max(nilai) - min(nilai)) / (2.0 * med) * 100.0 if med > 0 else float("nan")


def jalankan(rocks=None, out_dir=None, **_):
    rocks = rocks or ["Bentheimer", "Parker"]
    print("=" * 82)
    print("LANGKAH 07: REKAPITULASI TIGA SIFAT, SIFAT EFEKTIF, KONVERGENSI, BIAYA")
    print("=" * 82, flush=True)
    semua, sim_semua = {}, {}
    for kunci in rocks:
        dirb = dapatkan_subdirektori_batuan(out_dir, kunci)
        sim_dir = os.path.join(dirb, "simulasi_multiresolusi")
        rentang = baca_json(wajib(os.path.join(dirb, "rentang_segmentasi_acuan.json"), "02"))
        multi = {}
        with open(wajib(os.path.join(dirb, "tabel_multiresolusi.csv"), "05"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["batuan"] == kunci:
                    multi[(int(r["tingkat"]), r["cara"])] = r

        sap = rentang["sapuan_ambang"]
        phi0 = float(multi[(0, "resolusi_penuh")]["phi_tot"])
        sv0 = float(multi[(0, "resolusi_penuh")]["Sv_um1"])
        pita = {"phi": setengah_lebar([b["phi_tot"] for b in sap], phi0),
                "Sv": setengah_lebar([b["Sv_um1"] for b in sap], sv0),
                "k": float(rentang["setengah_lebar_persen"])}

        sim = {}
        for t, kode, dx, daftar in TINGKAT:
            for cara in daftar:
                for a in ARAH:
                    j = os.path.join(sim_dir, f"{kunci}_{kode}_{cara}_{a}.json")
                    sim[(t, cara, a)] = baca_json(j) if os.path.exists(j) else {}

        def k_arah(t, cara):
            return {a: float(sim[(t, cara, a)].get("K_mD", float("nan"))) for a in ARAH}

        k0 = k_arah(0, "resolusi_penuh")
        k0g = geo([k0[a] for a in ARAH])

        tiga, efektif = [], []
        for t, kode, dx, daftar in TINGKAT:
            for cara in daftar:
                m = multi.get((t, cara), {})
                phi = float(m.get("phi_tot", "nan"))
                sv = float(m.get("Sv_um1", "nan"))
                k = k_arah(t, cara)
                kg = geo([k[a] for a in ARAH])
                d = {"phi": (phi / phi0 - 1) * 100, "Sv": (sv / sv0 - 1) * 100,
                     "k": (kg / k0g - 1) * 100 if k0g > 0 else float("nan")}
                tiga.append({
                    "batuan": kunci, "tingkat": kode, "dx_um": dx, "cara": cara,
                    "phi_tot": phi, "Sv_um1": sv, "k_geo_mD": kg,
                    "simpangan_phi_persen": d["phi"], "simpangan_Sv_persen": d["Sv"], "simpangan_k_persen": d["k"],
                    "dalam_pita_phi": abs(d["phi"]) <= pita["phi"], "dalam_pita_Sv": abs(d["Sv"]) <= pita["Sv"],
                    "dalam_pita_k": bool(np.isfinite(d["k"]) and abs(d["k"]) <= pita["k"]),
                })
                tau = {a: float(sim[(t, cara, a)].get("diagnostik", {}).get("tau_u", float("nan"))) for a in ARAH}
                konv = all(bool(sim[(t, cara, a)].get("konvergen", False)) for a in ARAH)
                efektif.append({
                    "batuan": kunci, "tingkat": kode, "dx_um": dx, "cara": cara,
                    "kx_mD": k["x"], "ky_mD": k["y"], "kz_mD": k["z"], "k_geo_mD": kg,
                    "tau_u_x": tau["x"], "tau_u_y": tau["y"], "tau_u_z": tau["z"],
                    "tau_u_rerata": float(np.nanmean(list(tau.values()))) if any(np.isfinite(list(tau.values()))) else float("nan"),
                    "Ma_maks": max((float(sim[(t, cara, a)].get("diagnostik", {}).get("Ma", 0.0)) for a in ARAH), default=0.0),
                    "Re_pori_maks": max((float(sim[(t, cara, a)].get("diagnostik", {}).get("Re_pori", 0.0)) for a in ARAH), default=0.0),
                    "konvergen_xyz": konv,
                })

        runtuh = {}
        for cara in ["majority", "block_avg"]:
            runtuh[cara] = {}
            for sifat in ("phi", "Sv", "k"):
                lvl = "tidak runtuh sampai T3"
                for r in [x for x in tiga if x["cara"] == cara]:
                    if not r[f"dalam_pita_{sifat}"]:
                        lvl = f"{r['tingkat']} ({r['dx_um']:g} um)"
                        break
                runtuh[cara][sifat] = lvl

        hasil = {"batuan": kunci, "pita_segmentasi_persen": pita, "tingkat_runtuh": runtuh,
                 "perjanjian_perataan": ("k dari kecepatan Darcy = rerata u searah aliran atas seluruh volume "
                                         "sampel; satu angka pengganti = rerata geometrik kx, ky, kz (P11); "
                                         "tau_u = sum|u| / sum u_aliran atas simpul pori sampel (P11)"),
                 "tabel_tiga_sifat": tiga, "tabel_sifat_efektif": efektif}
        semua[kunci] = hasil

        print(f"\n>>> {kunci}: pita segmentasi phi +/-{pita['phi']:.2f}%, Sv +/-{pita['Sv']:.2f}%, k +/-{pita['k']:.2f}%")
        print(f"    {'Tingkat':<5} {'Cara':<16} {'d phi':>8} {'d Sv':>8} {'d k_geo':>9} {'tau_u':>7}")
        for r, e in zip(tiga, efektif):
            tanda = lambda ok: " " if ok else "*"
            print(f"    {r['tingkat']:<5} {NAMA_CARA[r['cara']]:<16} {r['simpangan_phi_persen']:+7.1f}%{tanda(r['dalam_pita_phi'])}"
                  f"{r['simpangan_Sv_persen']:+7.1f}%{tanda(r['dalam_pita_Sv'])}{r['simpangan_k_persen']:+8.1f}%{tanda(r['dalam_pita_k'])}"
                  f"{e['tau_u_rerata']:7.3f}{'' if e['konvergen_xyz'] else '  (belum konvergen)'}")
        print("    (* = di luar pita segmentasi sifat itu)")
        for cara, v in runtuh.items():
            print(f"    Tingkat runtuh [{NAMA_CARA[cara]}]: phi {v['phi']} | Sv {v['Sv']} | k {v['k']}")

        ang = anggaran_ketidakpastian(kunci, dirb, sim, rentang, float(tiga[0]["k_geo_mD"]))
        hasil["anggaran_ketidakpastian"] = ang
        print(f"    Anggaran ketidakpastian k T0 (P13, dalam ln k):")
        for b in ang["komponen"]:
            print(f"      {b['komponen']:<32} u = {b['u_persen']:6.2f} %  sumbangan ragam {b['sumbangan_ragam_persen']:5.1f} %  [{b['dasar']}]")
        print(f"      gabungan u_c = {100*ang['u_c_lnk']:.1f} %, U = {100*ang['U_lnk']:.1f} % -> k = {ang['k0_geo_mD']:.0f} mD, "
              f"selang 95 % {ang['selang95_mD'][0]:.0f} sampai {ang['selang95_mD'][1]:.0f} mD; "
              f"bias diskretisasi +{ang['bias_diskretisasi_persen']:.1f} %")
        tulis_csv(os.path.join(dirb, "tabel_anggaran_ketidakpastian.csv"),
                  [dict(batuan=kunci, **b) for b in ang["komponen"]])
        tulis_csv(os.path.join(dirb, "tabel_komparasi_tiga_sifat.csv"), tiga)
        tulis_csv(os.path.join(dirb, "tabel_sifat_efektif.csv"), efektif)
        with open(os.path.join(dirb, "rekapitulasi_akhir_s4.json"), "w", encoding="utf-8") as f:
            json.dump(hasil, f, indent=2)
        g = os.path.join(dirb, "grafik")
        plot_tiga_sifat(hasil, os.path.join(g, "07a_komparasi_tiga_sifat.png"))
        plot_riwayat([(kunci, sim)], os.path.join(g, "07b_riwayat_konvergensi_lbm.png"))
        sim_semua[kunci] = sim
        plot_biaya(kunci, sim, tiga, pita["k"], os.path.join(g, "07c_biaya_komputasi_dan_pareto.png"))

    if len(rocks) > 1:
        plot_tiga_sifat([semua[k] for k in rocks], os.path.join(out_dir, "grafik", "07a_komparasi_tiga_sifat.png"))
        plot_riwayat([(k, sim_semua[k]) for k in rocks], os.path.join(out_dir, "grafik", "07b_riwayat_konvergensi_lbm.png"))
        tulis_csv(os.path.join(out_dir, "tabel_komparasi_tiga_sifat.csv"), [r for k in rocks for r in semua[k]["tabel_tiga_sifat"]])
        tulis_csv(os.path.join(out_dir, "tabel_sifat_efektif.csv"), [r for k in rocks for r in semua[k]["tabel_sifat_efektif"]])
        with open(os.path.join(out_dir, "rekapitulasi_komparasi_lintas_batuan.json"), "w", encoding="utf-8") as f:
            json.dump(semua, f, indent=2)
    print("\n" + "=" * 82)
    print("SELESAI LANGKAH 07.")
    print("=" * 82)
    return semua


def tulis_csv(jalur, baris):
    if not baris:
        return
    with open(jalur, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)



def anggaran_ketidakpastian(kunci, dirb, sim, rentang, k0_geo):
    """Luaran 7: anggaran ketidakpastian k T0 menurut P13 (sel 'ANGGARAN KETIDAKPASTIAN').
    Semua komponen dalam ln k (ketidakpastian bekerja nisbi), digabung kuadrat: u_c = sqrt(sum u_i^2),
    U = 2 u_c, selang k0 exp(+/-U). Diskretisasi dicatat sebagai BIAS (koreksi galat resolusi, P16), bukan ragam."""
    import glob as _glob
    baris = []
    # 1) ambang segmentasi, tipe B: kepekaan di titik kerja x setengah selang / sqrt(3) (P13 Contoh Soal 1)
    setengah = max(abs(d) for d in rentang.get("delta_voxel", [0.5]))
    kep = abs(float(rentang["kepekaan_dlnk_ddelta_per_voxel"]))
    baris.append({"komponen": "ambang segmentasi", "tipe": "B",
                  "u_lnk": kep * setengah / math.sqrt(3.0),
                  "dasar": f"|d ln k/d delta| = {kep:.3f}/voxel x {setengah:g} voxel / sqrt(3) (P13)"})
    # 2) keterwakilan sampel: simpangan baku ln k antarsubvolume terbesar dari Langkah 04 (P13 u_sub)
    rev = {}
    for j in _glob.glob(os.path.join(dirb, "volume_elementer", f"{kunci}_L*_s*_x.json")):
        d = baca_json(j)
        if d.get("K_mD", 0) > 0:
            rev.setdefault(int(d.get("sisi_voxel", os.path.basename(j).split("_L")[1].split("_")[0])), []).append(d["K_mD"])
    if rev:
        L = max(rev)
        u_sub = float(np.log(rev[L]).std(ddof=1)) if len(rev[L]) > 1 else float("nan")
        baris.append({"komponen": "keterwakilan sampel (subvolume)", "tipe": "A", "u_lnk": u_sub,
                      "dasar": f"simpangan baku ln k, {len(rev[L])} subvolume {L}^3 arah x (Langkah 04, P13 u_sub)"})
    # 3) kriteria konvergensi: selisih K saat kriteria lama acuan (sisa < 5e-5, 2x) vs akhir (P7/P11), batas -> /sqrt(3)
    dmax = 0.0
    for a in ARAH:
        r = sim[(0, "resolusi_penuh", a)].get("riwayat", [])
        hit, lama = 0, None
        for v in r[1:]:
            hit = hit + 1 if v[2] < 5e-5 else 0
            if hit >= 2:
                lama = v[1]
                break
        if lama and r:
            dmax = max(dmax, abs(lama / r[-1][1] - 1))
    baris.append({"komponen": "kriteria konvergensi", "tipe": "B", "u_lnk": dmax / math.sqrt(3.0),
                  "dasar": f"selisih K kriteria lama (sisa<5e-5) vs P7/P11 maks {100*dmax:.3f} %, / sqrt(3)"})
    baris.append({"komponen": "derau citra", "tipe": "-", "u_lnk": float("nan"),
                  "dasar": "tidak dapat dinilai: citra sumber sudah biner (tanpa aras keabuan)"})
    uc = math.sqrt(sum(b["u_lnk"] ** 2 for b in baris if np.isfinite(b["u_lnk"])))
    for b in baris:
        b["u_persen"] = 100 * b["u_lnk"] if np.isfinite(b["u_lnk"]) else float("nan")
        b["sumbangan_ragam_persen"] = 100 * b["u_lnk"] ** 2 / uc ** 2 if np.isfinite(b["u_lnk"]) and uc > 0 else float("nan")
    U = 2 * uc
    # bias diskretisasi dari verifikasi (Langkah 00): galat saluran persegi pada lebar = d_t/dx voxel, orde teramati
    bias = float("nan")
    jv = os.path.join(os.path.dirname(dirb), "verifikasi", "verifikasi_pemecah.json")
    if os.path.exists(jv):
        v = baca_json(jv)
        sp = [k for k in v["kasus"] if k["kasus"] == "saluran_persegi"]
        p = v["ringkasan"]["saluran_persegi"]["orde_teramati"]
        acuan = max(sp, key=lambda k: k["ukuran_voxel"])
        nis = float(baca_json(os.path.join(dirb, "karakteristik_leher_pori.json"))["dt_um"]) / 2.25
        bias = abs(acuan["galat_nisbi"]) * (acuan["ukuran_voxel"] / nis) ** p
    return {"komponen": baris, "u_c_lnk": uc, "U_lnk": U, "k0_geo_mD": k0_geo,
            "selang95_mD": [k0_geo * math.exp(-U), k0_geo * math.exp(U)],
            "bias_diskretisasi_persen": 100 * bias,
            "catatan": "metode segmentasi tidak dimasukkan terpisah karena berkorelasi dengan ambang (P13)"}

def plot_tiga_sifat(hasil, out_img):
    """Luaran 13: simpangan phi, Sv, k terhadap T0, masing-masing dengan pita rentang segmentasinya sendiri.
    Lingkaran = tingkat pertama yang keluar dari pita (runtuh) untuk cara itu."""
    from program import gaya_grafik as G
    daftar = hasil if isinstance(hasil, list) else [hasil]          # satu baris panel per batuan
    fig, semua_ax = plt.subplots(len(daftar), 3, figsize=(13.5, 3.9 * len(daftar)), squeeze=False)
    for baris, hasil in zip(semua_ax, daftar):
        _panel_tiga_sifat(G, baris, hasil, judul_batuan=len(daftar) > 1)
    if len(daftar) == 1:
        fig.suptitle(f"{daftar[0]['batuan']}: tiga sifat terhadap rentang segmentasinya masing-masing",
                     fontweight="bold", fontsize=10.5)
    G.simpan(fig, out_img)


def _panel_tiga_sifat(G, axes, hasil, judul_batuan=False):
    pita = hasil["pita_segmentasi_persen"]
    t0 = [r for r in hasil["tabel_tiga_sifat"] if r["cara"] == "resolusi_penuh"]
    for ax, sifat in zip(axes, ("phi", "Sv", "k")):
        G.pita(ax, -pita[sifat], pita[sifat], label=f"rentang segmentasi ±{G.angka(pita[sifat], 1)} %")
        G.garis_acuan(ax, y=0, gaya="-", warna=G.TINTA_3, lw=0.8)
        for cara in ("majority", "block_avg"):
            sub = t0 + [r for r in hasil["tabel_tiga_sifat"] if r["cara"] == cara]
            x = [r["dx_um"] for r in sub]
            y = [r[f"simpangan_{sifat}_persen"] for r in sub]
            j = next((i for i, r in enumerate(sub) if not r[f"dalam_pita_{sifat}"]), None)
            ket = f" (runtuh {sub[j]['tingkat']})" if j is not None else " (tidak runtuh)"
            ax.plot(x, y, "-", marker=G.PENANDA_CARA[cara], color=G.CARA[cara], label=G.NAMA_CARA[cara] + ket)
            if j is not None:            # lingkaran = tingkat pertama di luar pita; keterangannya di legenda
                ax.plot(x[j], y[j], "o", ms=13, mfc="none", mec=G.CARA[cara], mew=1.2)
        G.sumbu_dx(ax)
        G.koma(ax, "y")
        ax.set_title(f"{hasil['batuan']}: {G.NAMA_SIFAT[sifat]}" if judul_batuan else G.NAMA_SIFAT[sifat])
        ax.legend(loc="best", fontsize=7.5)
    axes[0].set_ylabel("Simpangan terhadap T0 (%)")


def _teks_tunak(kasus, G):
    """Teks legenda: langkah saat kondisi tunak tercapai, atau langkah terakhir bila belum tunak."""
    n = G.angka(int(kasus.get("langkah", 0)), 0)
    return f"tunak {n} langkah" if kasus.get("konvergen") else f"BELUM tunak ({n} langkah)"


def plot_riwayat(daftar, out_img, sim=None):
    """Luaran 4: riwayat sisa |dK/K| (garis utuh) dan simpangan fluks antaririsan (putus-putus) per arah.
    daftar = [(batuan, sim), ...] -> satu baris per batuan (gabungan lintas batuan); bentuk lama
    plot_riwayat(batuan, out_img, sim) tetap diterima."""
    from program import gaya_grafik as G
    if isinstance(daftar, str):
        daftar = [(daftar, sim)]
    fig, axes = plt.subplots(len(daftar), 2, figsize=(12, 4.3 * len(daftar)), sharey=True, squeeze=False)
    ramp = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}       # biru berurutan T1 -> T3 (tingkat = ordinal)
    for baris, (kunci, sim) in zip(axes, daftar):
        for a in ARAH:
            kasus = sim[(0, "resolusi_penuh", a)]
            r = kasus.get("riwayat", [])
            if r:
                baris[0].plot([v[0] for v in r], [max(v[2], 1e-12) for v in r], "-", color=G.ARAH[a],
                              label=f"arah {a}: sisa, {_teks_tunak(kasus, G)}")
                if len(r[0]) > 5:
                    baris[0].plot([v[0] for v in r], [max(v[5], 1e-12) for v in r], "--", color=G.ARAH[a], lw=1.0,
                                  label=f"arah {a}: simpangan fluks")
        for t, kode, dx, _ in TINGKAT[1:]:
            for cara, gaya in (("majority", "-"), ("block_avg", "--")):
                kasus = sim[(t, cara, "x")]
                r = kasus.get("riwayat", [])
                if r:
                    baris[1].plot([v[0] for v in r], [max(v[2], 1e-12) for v in r], gaya, color=ramp[t], lw=1.4,
                                  label=f"{kode} {G.NAMA_CARA[cara].lower()}: {_teks_tunak(kasus, G)}")
        for ax, judul in zip(baris, ("resolusi penuh T0", "tingkat diturunkan, arah x")):
            ax.set_yscale("log")
            G.log_pangkat(ax, "y")
            for nilai, teks in ((TOL, "ambang sisa $10^{-6}$"), (TOL_FLUKS, "ambang fluks $10^{-3}$")):
                G.garis_acuan(ax, y=nilai)
                ax.annotate(teks, (1.0, nilai), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=G.TINTA_2)
            G.koma(ax, "x")
            ax.set_xlabel("Langkah waktu")
            ax.set_title(f"{kunci}: riwayat konvergensi, {judul}")
            ax.legend(fontsize=6.8, ncol=1 if ax is baris[1] else 2, loc="upper right")
        baris[0].set_ylabel("Sisa |ΔK/K|  /  simpangan fluks antaririsan")
    G.simpan(fig, out_img)


def plot_biaya(kunci, sim, tiga, pita_k, out_img):
    """Biaya: waktu LBM tiga arah (durasi produksi, perangkat menganggur) dan ketelitian terhadap biaya."""
    from program import gaya_grafik as G
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for cara in ("majority", "block_avg"):
        xs, durasi, simp, kode_t = [], [], [], []
        for t, kode, dx, daftar in TINGKAT:
            c = "resolusi_penuh" if t == 0 else cara
            d = sum(float(sim[(t, c, a)].get("durasi_detik", 0.0)) for a in ARAH)
            r = next(x for x in tiga if x["tingkat"] == kode and x["cara"] == c)
            xs.append(dx); durasi.append(max(d, 1e-3)); simp.append(abs(r["simpangan_k_persen"])); kode_t.append(kode)
        axes[0].plot(xs, durasi, "-", marker=G.PENANDA_CARA[cara], color=G.CARA[cara], label=G.NAMA_CARA[cara])
        axes[1].plot(durasi, simp, "-", marker=G.PENANDA_CARA[cara], color=G.CARA[cara], label=G.NAMA_CARA[cara])
        for x, y, k in zip(durasi, simp, kode_t):
            if k != "T0" or cara == "majority":
                G.label_ujung(axes[1], x, y, k, dx=5, dy=5, va="bottom", ukuran=7.5, warna=G.TINTA_2)
    G.sumbu_dx(axes[0])
    axes[0].set_yscale("log")
    G.log_angka(axes[0], "y")
    axes[0].set_ylabel("Waktu LBM tiga arah (detik)")
    axes[0].set_title(f"{kunci}: biaya komputasi per tingkat")
    G.garis_acuan(axes[1], y=pita_k, label=f"setengah lebar rentang segmentasi {G.angka(pita_k, 1)} %")
    axes[1].set_xscale("log")
    G.log_angka(axes[1], "x")
    G.koma(axes[1], "y")
    axes[1].set_xlabel("Waktu LBM tiga arah (detik)")
    axes[1].set_ylabel("|Simpangan $k_{geo}$| terhadap T0 (%)")
    axes[1].set_title("Ketelitian terhadap biaya")
    for ax in axes:
        ax.legend(fontsize=7.5)
    G.simpan(fig, out_img)
