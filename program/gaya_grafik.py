"""
program/gaya_grafik.py
======================
Gaya grafik bersama untuk semua langkah (GPU, CPU, inti), supaya seluruh gambar laporan seragam.

Aturan
  * Warna mengikuti ENTITAS, urutan tetap (palet tervalidasi buta warna, 3 slot pertama lolos semua pasangan):
      arah   x / y / z                  -> biru / oranye / aqua
      cara   suara terbanyak / rerata blok -> biru / oranye
      sifat  phi / Sv / k               -> biru / oranye / aqua
    Setiap seri juga punya bentuk penanda sendiri (pengodean kedua, tidak bergantung warna saja).
  * Garis acuan (ambang, Nyquist, pita segmentasi) memakai tinta netral abu-abu, bukan warna seri;
    warna status tidak dipakai untuk seri.
  * Satu sumbu y per panel (tidak ada sumbu ganda); dua besaran berbeda skala -> dua panel.
  * Bahasa Indonesia: koma desimal, satuan "µm", judul kalimat biasa.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator, NullLocator, LogLocator

# Palet kategorikal (slot 1-3 lolos semua pasangan, mode terang)
BIRU, ORANYE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, TINTA_3 = "#0b0b0b", "#52514e", "#8a8984"
KISI, PITA = "#e8e8e5", "#d9d9d5"

ARAH = {"x": BIRU, "y": ORANYE, "z": AQUA}
PENANDA_ARAH = {"x": "o", "y": "s", "z": "^"}
CARA = {"majority": BIRU, "block_avg": ORANYE, "resolusi_penuh": TINTA}
PENANDA_CARA = {"majority": "o", "block_avg": "s", "resolusi_penuh": "D"}
NAMA_CARA = {"resolusi_penuh": "Resolusi penuh", "majority": "Suara terbanyak", "block_avg": "Rerata blok"}
SIFAT = {"phi": BIRU, "Sv": ORANYE, "k": AQUA}
PENANDA_SIFAT = {"phi": "o", "Sv": "s", "k": "D"}
NAMA_SIFAT = {"phi": "Porositas φ", "Sv": "Luas spesifik Sv", "k": "Permeabilitas k"}
DX_TINGKAT = [2.25, 4.5, 9.0, 18.0]


def terapkan():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.facecolor": "white", "figure.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "text.color": TINTA, "axes.labelcolor": TINTA, "axes.edgecolor": TINTA_2,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.grid": True, "grid.color": KISI, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 5.5,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "axes.formatter.use_mathtext": True, "mathtext.default": "regular",
        "axes.unicode_minus": True,
    })


def angka(v, desimal=None):
    """Angka berformat Indonesia: koma desimal, titik ribuan (bilangan bulat besar)."""
    if desimal is None:
        s = f"{v:g}"
    else:
        s = f"{v:,.{desimal}f}"
    if "e" in s:
        return s.replace(".", ",")
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def _koma(x, _pos=None):
    if abs(x) >= 1000 and float(x).is_integer():
        return angka(x, 0).replace("-", "\u2212")
    s = f"{x:.6g}"
    if "e" in s:
        m, e = s.split("e")
        return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"
    return s.replace(".", ",").replace("-", "−")


def koma(ax, sumbu="xy"):
    """Label sumbu linear dengan koma desimal."""
    for s in sumbu:
        a = ax.xaxis if s == "x" else ax.yaxis
        a.set_major_formatter(FuncFormatter(_koma))


def log_pangkat(ax, sumbu="y"):
    """Sumbu log dengan label 10^n saja (tanpa label minor yang bertumpuk)."""
    for s in sumbu:
        a = ax.xaxis if s == "x" else ax.yaxis
        a.set_major_formatter(FuncFormatter(lambda v, _p: f"$10^{{{int(round(__import__('math').log10(v)))}}}$"))
        a.set_minor_formatter(NullFormatter())


def log_angka(ax, sumbu="y"):
    """Sumbu log dengan label angka biasa (0,1 / 1 / 10 / 100 / 1.000), tanpa label minor."""
    import math
    for s in sumbu:
        a = ax.xaxis if s == "x" else ax.yaxis
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(FuncFormatter(lambda v, _p: angka(v, max(0, -int(math.floor(math.log10(v))))) if v > 0 else "0"))
        a.set_minor_locator(LogLocator(base=10, subs="auto"))
        a.set_minor_formatter(NullFormatter())


def sumbu_dx(ax, dx=DX_TINGKAT, tingkat=True, label="Ukuran voxel Δx (µm)"):
    """Sumbu x Δx berskala log-2 dengan tik tepat di keempat tingkat: '2,25\\nT0'."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(dx))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{angka(v)}\nT{i}" if tingkat else angka(v) for i, v in enumerate(dx)])
    ax.set_xlim(dx[0] / 1.25, dx[-1] * 1.25)
    ax.set_xlabel(label)


def garis_acuan(ax, y=None, x=None, label=None, gaya="--", warna=TINTA_2, lw=1.0):
    if y is not None:
        return ax.axhline(y, ls=gaya, color=warna, lw=lw, label=label, zorder=1)
    return ax.axvline(x, ls=gaya, color=warna, lw=lw, label=label, zorder=1)


def pita(ax, bawah, atas, label=None, warna=PITA, alpha=0.55):
    return ax.axhspan(bawah, atas, color=warna, alpha=alpha, lw=0, label=label, zorder=0)


def label_ujung(ax, x, y, teks, warna=TINTA, dx=4, dy=0, ha="left", va="center", ukuran=8):
    """Label langsung di ujung seri (teks bertinta netral)."""
    ax.annotate(teks, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=ukuran, color=warna)


def simpan(fig, jalur):
    os.makedirs(os.path.dirname(jalur), exist_ok=True)
    fig.tight_layout()
    fig.savefig(jalur, bbox_inches="tight")
    plt.close(fig)


terapkan()
