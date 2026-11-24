#!/usr/bin/env python3
"""
program/inti/inti_00_verifikasi_pemecah.py
=====================================
Langkah 00 (luaran wajib 5): verifikasi pemecah LBM D3Q19 terhadap penyelesaian analitik Stokes
(Pertemuan 9, "Solusi Analitik sebagai Jangkar" dan "Orde Konvergensi").

Prinsip P9: jangkar harus dijalankan dengan pemecah dan pengaturan yang SAMA dengan
simulasi batuan. Karena itu modul ini memanggil langsung `simulasikan_lbm` dari
program/<backend>/<backend>_solver_lbm.py dengan pengaturan yang sama persis:
BGK tau = 1,0, beda kerapatan Zou-He drho = 0,002, lapisan penyangga 4 voxel,
dinding samping tanpa geser (bounce-back setengah jalan), dan rumus permeabilitas
K = nu * q * L / dP yang dipakai pada batuan.

Dua penampang analitik (P9):
1. Saluran persegi bersisi s voxel fluida (dinding sejajar kisi, tanpa galat geometri).
   K_eksak (atas luas fluida) = C * s^2, C = (1/12) * [1 - (192/pi^5) sum_{n ganjil} tanh(n*pi/2)/n^5]
                                                 = 0,035144 (deret penuh, P9 Contoh Soal 1).
2. Pipa bundar berdiameter D voxel (dinding bertangga, memuat galat geometri).
   K_eksak atas luas kotak penampang = pi * R^4 / (8 * A_kotak), R = D/2.

Keluaran: galat nisbi tiap resolusi, orde teramati p dari kemiringan log-log,
ekstrapolasi Richardson dan Grid Convergence Index (GCI) pada tiga resolusi terhalus.

Pemakaian:
    ./program/gpu/gpu_00_verifikasi_pemecah.py
    ./program/cpu/cpu_00_verifikasi_pemecah.py --sisi 4 8 16 32
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
_VENV_DIR = os.path.join(ROOT, ".venv")
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

import csv
import json
import math
import time
import argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TAU = 1.0
DRHO = 0.002
BUFFER = 4
PANJANG_SAMPEL = 32          # voxel searah aliran
FS = 1.25                    # faktor keamanan GCI untuk tiga resolusi (Roache, P9)


def muat_solver(backend):
    jalur = os.path.join(ROOT, "program", backend, f"{backend}_solver_lbm.py")
    spec = importlib.util.spec_from_file_location(f"solver_lbm_{backend}", jalur)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def konstanta_saluran_persegi(n_suku=50):
    """C pada K = C * s^2 untuk saluran persegi (deret Fourier, P9 Contoh Soal 1)."""
    jumlah = sum(math.tanh(n * math.pi / 2.0) / n ** 5 for n in range(1, 2 * n_suku, 2))
    return (1.0 - 192.0 / math.pi ** 5 * jumlah) / 12.0


def penampang(jenis, ukuran):
    """Masker fluida 2D (ukuran x ukuran) untuk penampang tegak lurus aliran."""
    if jenis == "saluran_persegi":
        return np.ones((ukuran, ukuran), dtype=bool)
    # pipa bundar: pusat sel di dalam lingkaran berjari-jari D/2
    c = (ukuran - 1) / 2.0
    y, z = np.mgrid[0:ukuran, 0:ukuran]
    return (y - c) ** 2 + (z - c) ** 2 <= (ukuran / 2.0) ** 2


def k_eksak(jenis, ukuran, masker):
    if jenis == "saluran_persegi":
        return konstanta_saluran_persegi() * ukuran ** 2      # atas luas fluida = luas kotak
    R = ukuran / 2.0
    return math.pi * R ** 4 / (8.0 * masker.size)            # atas luas kotak (sama dengan cara batuan)


def bangun_domain(masker, panjang=PANJANG_SAMPEL, buffer_vox=BUFFER):
    """Domain bergaya batuan: penyangga + sampel + penyangga, dinding padat 1 lapis di sisi samping."""
    ny, nz = masker.shape
    nx = panjang + 2 * buffer_vox
    fluid = np.zeros((nx, ny + 2, nz + 2), dtype=bool)
    fluid[:, 1:-1, 1:-1] = masker[None, :, :]
    roi = (slice(buffer_vox, buffer_vox + panjang), slice(1, -1), slice(1, -1))
    bidang_tekanan = [buffer_vox - 1, buffer_vox + panjang]
    return fluid, roi, bidang_tekanan


def orde_teramati(h, galat):
    """Kemiringan log|galat| terhadap log h (h = 1/ukuran) dengan kuadrat terkecil."""
    x = np.log(np.asarray(h, float))
    y = np.log(np.abs(np.asarray(galat, float)))
    return float(np.polyfit(x, y, 1)[0])


def richardson_gci(ukuran, k):
    """Richardson dan GCI pada tiga resolusi terhalus dengan nisbah penghalusan tetap (P9)."""
    if len(ukuran) < 3:
        return None
    (s1, s2, s3), (f1, f2, f3) = ukuran[-3:][::-1], k[-3:][::-1]   # 1 = terhalus
    r = s1 / s2
    if abs(s2 / s3 - r) > 1e-9 or (f3 - f2) == 0 or (f2 - f1) == 0:
        return None
    nisbah = (f3 - f2) / (f2 - f1)
    if nisbah <= 0:
        return {"r": r, "p": float("nan"), "catatan": "tidak monoton, di luar wilayah asimtotik"}
    p = math.log(nisbah) / math.log(r)
    if p < 0.5:
        return {"r": r, "p": p, "catatan": "orde tidak wajar, di luar wilayah asimtotik (galat geometri tangga)"}
    f_ext = f1 + (f1 - f2) / (r ** p - 1.0)
    e21 = abs((f1 - f2) / f1)
    gci = FS * e21 / (r ** p - 1.0)
    return {"r": r, "p": p, "k_ekstrapolasi": f_ext, "GCI_halus_persen": 100.0 * gci}


def jalankan(backend="gpu", out_dir=None, sisi=(4, 8, 16, 32), diameter=(8, 16, 32), verbose=True):
    solver = muat_solver(backend)
    out_dir = out_dir or os.path.join(ROOT, "hasil", backend, "verifikasi")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 82)
    print(f"LANGKAH 00: VERIFIKASI PEMECAH LBM D3Q19 TERHADAP PENYELESAIAN ANALITIK [{backend.upper()}]")
    print(f"  * Pengaturan       : sama dengan simulasi batuan (BGK tau={TAU}, Zou-He drho={DRHO},")
    print(f"                       penyangga {BUFFER} voxel, dinding samping bounce-back setengah jalan)")
    print(f"  * Panjang sampel   : {PANJANG_SAMPEL} voxel searah aliran")
    print(f"  * Konstanta saluran: C = {konstanta_saluran_persegi():.6f} (P9: 0,035144)")
    print("=" * 82)

    kasus = [("saluran_persegi", s) for s in sisi] + [("pipa_bundar", d) for d in diameter]
    baris = []
    for jenis, ukuran in kasus:
        masker = penampang(jenis, ukuran)
        fluid, roi, bidang = bangun_domain(masker)
        label = f"{jenis}_{ukuran}"
        t0 = time.monotonic()
        res = solver.simulasikan_lbm(fluid, roi, bidang, tau=TAU, drho=DRHO, maks=400000,
                                     tol=1e-7, periksa=200, label=label, verbose=False)
        ke = k_eksak(jenis, ukuran, masker)
        galat = (res["k"] - ke) / ke
        riwayat = res["riwayat"]
        baris.append({
            "kasus": jenis, "ukuran_voxel": ukuran,
            "K_kisi_simulasi": res["k"], "K_kisi_eksak": ke, "galat_nisbi": galat,
            "langkah": res["langkah"], "konvergen": res["konvergen"],
            "residual_akhir": riwayat[-1][2] if riwayat else float("nan"),
            "durasi_detik": time.monotonic() - t0,
        })
        with open(os.path.join(out_dir, f"riwayat_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"kasus": label, "riwayat": riwayat}, f)
        if verbose:
            print(f"  {label:<22} K_sim={res['k']:.6e}  K_eksak={ke:.6e}  galat={100*galat:+8.3f}%  "
                  f"langkah={res['langkah']:>6}  konvergen={res['konvergen']}", flush=True)

    ringkasan = {}
    for jenis in ("saluran_persegi", "pipa_bundar"):
        sub = [b for b in baris if b["kasus"] == jenis]
        if len(sub) < 2:
            continue
        uk = [b["ukuran_voxel"] for b in sub]
        ks = [b["K_kisi_simulasi"] / b["ukuran_voxel"] ** 2 for b in sub]   # K/s^2 tak berdimensi
        ringkasan[jenis] = {
            "orde_teramati": orde_teramati([1.0 / u for u in uk], [b["galat_nisbi"] for b in sub]),
            "richardson_gci": richardson_gci(uk, ks),
            "galat_terhalus_persen": 100.0 * sub[-1]["galat_nisbi"],
        }

    with open(os.path.join(out_dir, "tabel_verifikasi.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)
    with open(os.path.join(out_dir, "verifikasi_pemecah.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": backend, "tau": TAU, "drho": DRHO, "buffer": BUFFER,
                   "panjang_sampel": PANJANG_SAMPEL, "kasus": baris, "ringkasan": ringkasan}, f, indent=2)

    plot_verifikasi(baris, os.path.join(out_dir, "00_verifikasi_pemecah.png"))

    print("-" * 82)
    for jenis, r in ringkasan.items():
        g = r["richardson_gci"]
        teks_gci = (f"Richardson p={g['p']:.2f}, GCI={g['GCI_halus_persen']:.3f}%"
                    if g and "GCI_halus_persen" in g else "Richardson tidak berlaku")
        print(f"  {jenis:<16}: orde teramati = {r['orde_teramati']:.2f} | galat terhalus = "
              f"{r['galat_terhalus_persen']:+.3f}% | {teks_gci}")
    print("-" * 82)
    print(f"[OK] Hasil verifikasi tersimpan di: {out_dir}/")
    return {"kasus": baris, "ringkasan": ringkasan}


def plot_verifikasi(baris, out_img):
    """Luaran 5: galat nisbi K terhadap solusi Stokes analitik vs ukuran penampang, dengan orde teramati."""
    from program import gaya_grafik as G
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    semua_x = []
    for jenis, warna, mk, nama in (("saluran_persegi", G.BIRU, "o", "Saluran persegi"),
                                   ("pipa_bundar", G.ORANYE, "s", "Pipa bundar (dinding bertangga)")):
        sub = [b for b in baris if b["kasus"] == jenis]
        if not sub:
            continue
        x = np.array([b["ukuran_voxel"] for b in sub], float)
        y = np.array([abs(b["galat_nisbi"]) * 100 for b in sub])
        semua_x += list(x)
        p_ord = orde_teramati([1.0 / u for u in x], [b["galat_nisbi"] for b in sub])   # sama dengan tabel
        ax.plot(x, y, "-", marker=mk, color=warna, label=f"{nama}, orde teramati {G.angka(p_ord, 2)}")
    xs = np.array([min(semua_x), max(semua_x)], float)
    ax.plot(xs, 10.0 * (xs / xs[0]) ** -2, "--", color=G.TINTA_2, lw=1.0, label="acuan kemiringan orde 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    tik = sorted(set(int(v) for v in semua_x))
    ax.xaxis.set_major_locator(FixedLocator(tik))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in tik])
    G.log_angka(ax, "y")
    ax.set_xlabel("Ukuran penampang (voxel)")
    ax.set_ylabel("|Galat nisbi K| terhadap analitik (%)")
    ax.set_title("Verifikasi pemecah terhadap solusi Stokes analitik")
    ax.legend(fontsize=7.5, loc="lower left")
    G.simpan(fig, out_img)


def main():
    p = argparse.ArgumentParser(description="Verifikasi pemecah LBM terhadap solusi analitik (P9)")
    p.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--sisi", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--diameter", nargs="+", type=int, default=[8, 16, 32])
    a = p.parse_args()
    jalankan(a.backend, a.out_dir, tuple(a.sisi), tuple(a.diameter))


if __name__ == "__main__":
    main()
