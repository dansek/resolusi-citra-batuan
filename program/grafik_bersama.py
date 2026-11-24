"""
program/grafik_bersama.py
=========================
Grafik langkah 01 dan 03 yang dipakai bersama oleh backend GPU dan CPU (satu sumber, tidak ada salinan ganda).
Gaya: program/gaya_grafik.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from program import gaya_grafik as G


def plot_konvergensi_t0(hasil_simulasi, rocks, out_img, tol=1e-6, tol_fluks=1e-3):
    """Luaran 4 (langkah 01): evolusi k per arah dan riwayat sisa / simpangan fluks, satu baris per batuan."""
    rocks = [r for r in rocks if r in hasil_simulasi]
    fig, axes = plt.subplots(len(rocks), 2, figsize=(12, 4.1 * len(rocks)), squeeze=False)
    for i, batuan in enumerate(rocks):
        d = hasil_simulasi[batuan]
        ax1, ax2 = axes[i]
        for a in ("x", "y", "z"):
            r = d["riwayat_konvergensi"].get(a)
            if not r or not r.get("steps"):
                continue
            status = "tunak" if r["konvergen"] else "BELUM tunak"
            ax1.plot(r["steps"], r["k_mD_riwayat"], "-", color=G.ARAH[a],
                     label=f"$k_{a}$ = {G.angka(r['K_mD'], 1)} mD ({G.angka(r['langkah'], 0)} langkah, {status})")
            if r.get("residual_riwayat"):
                ax2.plot(r["steps"], [max(v, 1e-12) for v in r["residual_riwayat"]], "-", color=G.ARAH[a],
                         label=f"arah {a}: sisa")
            if r.get("fluks_riwayat"):
                ax2.plot(r["steps"][-len(r["fluks_riwayat"]):], [max(v, 1e-12) for v in r["fluks_riwayat"]], "--",
                         color=G.ARAH[a], lw=1.0, label=f"arah {a}: simpangan fluks")
        kg = d.get("K_geometrik_mD", 0.0)
        if kg > 0:
            G.garis_acuan(ax1, y=kg, label=f"$k_{{geo}}$ = $\\sqrt[3]{{k_x k_y k_z}}$ = {G.angka(kg, 1)} mD")
        ax1.set_yscale("log")
        G.log_angka(ax1, "y")
        ax1.set_ylabel("Permeabilitas k (mD)")
        ax1.set_title(f"{batuan}: evolusi permeabilitas resolusi penuh ({d.get('N', 256)}³)")
        ax1.legend(fontsize=7.5, loc="best")
        ax2.set_yscale("log")
        G.log_pangkat(ax2, "y")
        for nilai, teks in ((tol, "ambang sisa $10^{-6}$"), (tol_fluks, "ambang fluks $10^{-3}$")):
            G.garis_acuan(ax2, y=nilai)
            ax2.annotate(teks, (1.0, nilai), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                         textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=G.TINTA_2)
        ax2.set_ylabel("Sisa |ΔK/K|  /  simpangan fluks antaririsan")
        ax2.set_title(f"{batuan}: kriteria kondisi tunak")
        ax2.legend(fontsize=7, ncol=2, loc="upper right")
        for ax in (ax1, ax2):
            G.koma(ax, "x")
            ax.set_xlabel("Langkah waktu")
    G.simpan(fig, out_img)


def plot_leher_pori(hasil_leher_pori, data_histogram, out_img, dx_um=2.25):
    """Luaran 1 (langkah 03): sebaran lebar leher (median skeleton) dengan tiga takrif d_t, dan kurva granulometri.
    Satu baris per batuan; garis acuan bertinta netral dengan label langsung."""
    rocks = list(hasil_leher_pori)
    fig, axes = plt.subplots(len(rocks), 2, figsize=(12, 4.0 * len(rocks)), squeeze=False)
    for i, kunci in enumerate(rocks):
        d = hasil_leher_pori[kunci]
        ax1, ax2 = axes[i]
        data = np.asarray(data_histogram.get(kunci, []), float)
        atas = max(float(data.max()) if data.size else 0.0, d["dt_granulometri_um"], d.get("dt_um", 0.0)) * 1.05
        bins = np.arange(0, atas + 1.5, 1.5)          # semua data masuk (tidak ada yang terpotong)
        if kunci in data_histogram and len(data_histogram[kunci]):
            ax1.hist(data_histogram[kunci], bins=bins, density=True, color=G.BIRU, alpha=0.85,
                     edgecolor="white", linewidth=0.8, label="lebar leher pada skeleton")
        garis = [(d.get("dt_um"), "-", 1.8, G.TINTA, f"$d_t$ penentu = {G.angka(d.get('dt_um', 0), 1)} µm"),
                 (d["d_skeleton"]["d_median_um"], "--", 1.0, G.TINTA_2,
                  f"median skeleton = {G.angka(d['d_skeleton']['d_median_um'], 1)} µm"),
                 (d["dt_granulometri_um"], ":", 1.2, G.TINTA_2,
                  f"granulometri 2·$r_{{50}}$ = {G.angka(d['dt_granulometri_um'], 1)} µm"),
                 (2 * dx_um, "-.", 1.0, G.TINTA_3, f"batas Nyquist T0 (2Δx = {G.angka(2 * dx_um)} µm)")]
        for x, ls, lw, w, lbl in garis:
            if x:
                ax1.axvline(x, ls=ls, lw=lw, color=w, label=lbl)
        G.koma(ax1)
        ax1.set_xlabel("Diameter leher pori d (µm)")
        ax1.set_ylabel("Kerapatan peluang (1/µm)")
        ax1.set_title(f"{kunci}: sebaran lebar leher pori")
        ax1.legend(fontsize=7.5, loc="upper right")
        gran = d["kurva_granulometri"]
        ax2.plot(gran["jari_um"], gran["fraksi_volume"], "-o", ms=3.5, color=G.BIRU, label="fraksi pori termasuki")
        G.garis_acuan(ax2, y=0.5)
        G.garis_acuan(ax2, x=d["r50_granulometri_um"], gaya=":")
        G.label_ujung(ax2, d["r50_granulometri_um"], 0.5, f"$r_{{50}}$ = {G.angka(d['r50_granulometri_um'], 2)} µm",
                      dx=6, dy=8, va="bottom", warna=G.TINTA_2)
        G.koma(ax2)
        ax2.set_xlabel("Jari-jari bola penguji r (µm)")
        ax2.set_ylabel("Fraksi ruang pori yang termasuki")
        ax2.set_title(f"{kunci}: granulometri (tiruan intrusi raksa, pembanding)")
    G.simpan(fig, out_img)
