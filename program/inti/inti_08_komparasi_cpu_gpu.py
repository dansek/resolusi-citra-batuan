#!/usr/bin/env python3
"""
program/inti/inti_08_komparasi_cpu_gpu.py
=========================================
Langkah 08: komparasi waktu, memori, dan hasil antara backend GPU (CuPy) dan CPU (Numba) dari run PRODUKSI.

Syarat adil (pengganti benchmark terkendali): kedua backend dijalankan berurutan, satu kasus per waktu, dan setiap
berkas kasus mencatat `kondisi_perangkat` (program/kondisi_perangkat.py). Sepasang kasus dipakai bila
  * kedua backend punya berkasnya dan cacah langkahnya SAMA (pemecah identik -> kriteria tercapai di langkah sama),
  * kedua `kondisi_perangkat.menganggur` bernilai True; pasangan lain dicatat 'tidak adil' dan dikeluarkan.
Besaran per tingkat (gabungan 3 arah dan kedua cara penurunan): detik, detik per 1.000 langkah, MLUPS (P7),
percepatan t_CPU / t_GPU, memori (taksiran P7 kisi penuh N^3 x 19 x 8 x 2, larik terukur, puncak kolam GPU,
RSS puncak proses CPU), dan selisih K GPU-CPU (validasi silang).

  python program/inti/inti_08_komparasi_cpu_gpu.py --rocks Bentheimer BSG
"""

import os
import sys
import csv
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TINGKAT = [(0, "T0", 2.25), (1, "T1", 4.5), (2, "T2", 9.0), (3, "T3", 18.0)]


def _baca(j):
    with open(j, encoding="utf-8") as f:
        return json.load(f)


def kumpulkan(base_out_dir, rock):
    """Pasangan kasus multiresolusi GPU-CPU satu batuan, dipilah: dipakai / tidak adil / beda langkah."""
    pasangan, tidak_adil, beda_langkah = [], [], []
    for jg in sorted(glob.glob(os.path.join(base_out_dir, "gpu", rock, "simulasi_multiresolusi", "*.json"))):
        jc = os.path.join(base_out_dir, "cpu", rock, "simulasi_multiresolusi", os.path.basename(jg))
        if not os.path.exists(jc):
            continue
        g, c = _baca(jg), _baca(jc)
        if int(g.get("langkah", 0)) == 0:            # domain tak tembus: tidak ada simulasi
            continue
        nama = os.path.basename(jg)[:-5]
        if g.get("langkah") != c.get("langkah"):
            beda_langkah.append((nama, g.get("langkah"), c.get("langkah")))
            continue
        if not ((g.get("kondisi_perangkat") or {}).get("menganggur") is True and
                (c.get("kondisi_perangkat") or {}).get("menganggur") is True):
            tidak_adil.append(nama)
            continue
        pasangan.append((int(g["tingkat"]), g, c))
    return pasangan, tidak_adil, beda_langkah


def _mem(kasus, kunci):
    return max(float((d.get("memori") or {}).get(kunci) or 0.0) for d in kasus)


def rekap_tingkat(pasangan):
    baris = []
    for t, kode, dx in TINGKAT:
        sub = [(g, c) for tt, g, c in pasangan if tt == t]
        if not sub:
            continue
        gs, cs = [g for g, _ in sub], [c for _, c in sub]
        lgk = sum(g["langkah"] for g in gs)
        dg = sum(g["durasi_detik"] for g in gs)
        dc = sum(c["durasi_detik"] for c in cs)
        lu = sum(g["langkah"] * g.get("n_fluida", 0) for g in gs)
        baris.append({
            "tingkat": kode, "dx_um": dx, "kasus": len(sub), "langkah_total": lgk,
            "detik_gpu": dg, "detik_cpu": dc, "percepatan": dc / dg if dg > 0 else float("nan"),
            "detik_per_1000_langkah_gpu": 1000 * dg / lgk, "detik_per_1000_langkah_cpu": 1000 * dc / lgk,
            "mlups_gpu": lu / dg / 1e6 if dg > 0 else float("nan"),
            "mlups_cpu": lu / dc / 1e6 if dc > 0 else float("nan"),
            "memori_P7_kisi_penuh_gb": _mem(gs, "taksiran_P7_kisi_penuh_gb"),
            "memori_larik_utama_gb": _mem(gs, "larik_utama_gb"),
            "memori_kolam_gpu_puncak_gb": _mem(gs, "kolam_gpu_puncak_gb"),
            "rss_puncak_proses_cpu_gb": _mem(cs, "rss_puncak_proses_gb"),
            "selisih_K_maks_nisbi": max(abs(g["K_mD"] - c["K_mD"]) / max(g["K_mD"], 1e-300) for g, c in sub),
        })
    return baris


def plot(rekap, out_img):
    from program import gaya_grafik as G
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, (rock, b) in enumerate(rekap.items()):
        dx = [r["dx_um"] for r in b]
        ls = ["-", "--", ":"][i % 3]
        axes[0].plot(dx, [r["mlups_gpu"] for r in b], ls, marker="o", color=G.BIRU, label=f"GPU, {rock}")
        axes[0].plot(dx, [r["mlups_cpu"] for r in b], ls, marker="s", color=G.ORANYE, label=f"CPU, {rock}")
        axes[1].plot(dx, [r["percepatan"] for r in b], ls, marker="D", color=G.TINTA, label=rock)
        axes[2].plot(dx, [r["memori_P7_kisi_penuh_gb"] for r in b], ls, marker="^", color=G.TINTA_2,
                     label=f"taksiran P7 kisi penuh, {rock}")
        axes[2].plot(dx, [r["memori_larik_utama_gb"] for r in b], ls, marker="o", color=G.BIRU,
                     label=f"larik terukur (simpul fluida), {rock}")
    for ax, yl, jdl in ((axes[0], "MLUPS", "Laju pemecah"),
                        (axes[1], "Percepatan $t_{CPU}$ / $t_{GPU}$ (×)", "Percepatan GPU terhadap CPU"),
                        (axes[2], "Memori (GB)", "Memori satu kasus")):
        G.sumbu_dx(ax)
        if ax is axes[1]:                     # percepatan: rentang sempit, sumbu linear mulai 0
            ax.set_ylim(bottom=0)
            G.koma(ax, "y")
        else:
            ax.set_yscale("log")
            G.log_angka(ax, "y")
        ax.set_ylabel(yl)
        ax.set_title(jdl)
        ax.legend(fontsize=7)
    G.simpan(fig, out_img)


def jalankan_komparasi(base_out_dir, rocks, verbose=True):
    rekap, catatan = {}, {}
    for rock in rocks:
        if not all(os.path.isdir(os.path.join(base_out_dir, be, rock)) for be in ("gpu", "cpu")):
            continue
        pasangan, tidak_adil, beda = kumpulkan(base_out_dir, rock)
        catatan[rock] = {"kasus_dipakai": len(pasangan), "tidak_adil": tidak_adil, "langkah_berbeda": beda}
        if pasangan:
            rekap[rock] = rekap_tingkat(pasangan)
    if verbose:
        print("=" * 82)
        print("LANGKAH 08: KOMPARASI GPU vs CPU (durasi produksi, perangkat menganggur, langkah sama)")
        print("=" * 82)
        for rock, c in catatan.items():
            print(f"\n>>> {rock}: {c['kasus_dipakai']} pasangan dipakai; {len(c['tidak_adil'])} tidak adil "
                  f"(perangkat dipakai proses lain); {len(c['langkah_berbeda'])} beda langkah", flush=True)
            for r in rekap.get(rock, []):
                print(f"    {r['tingkat']}  GPU {r['mlups_gpu']:8.1f} MLUPS | CPU {r['mlups_cpu']:8.1f} MLUPS | "
                      f"percepatan {r['percepatan']:6.1f}x | memori P7 {r['memori_P7_kisi_penuh_gb']:.3f} GB, "
                      f"terukur {r['memori_larik_utama_gb']:.3f} GB | selisih K {r['selisih_K_maks_nisbi']:.1e}")
    if not rekap:
        if verbose:
            print("    Belum ada pasangan hasil GPU dan CPU yang adil; komparasi dilewati.")
        return None
    baris = [dict(batuan=rock, **r) for rock in rekap for r in rekap[rock]]
    with open(os.path.join(base_out_dir, "tabel_komparasi_cpu_gpu.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0]))
        w.writeheader()
        w.writerows(baris)
    with open(os.path.join(base_out_dir, "komparasi_cpu_gpu.json"), "w", encoding="utf-8") as f:
        json.dump({"rekap": rekap, "catatan": catatan}, f, indent=2)
    plot(rekap, os.path.join(base_out_dir, "grafik", "08_komparasi_cpu_gpu.png"))
    return rekap


def jalankan(rocks=None, out_dir=None, **_):
    """Pembungkus run.py: out_dir = <induk>/<backend>; komparasi membaca induknya."""
    base = os.path.dirname(os.path.normpath(out_dir)) if out_dir else os.path.join(ROOT, "hasil")
    return jalankan_komparasi(base, rocks or ["Bentheimer", "BSG"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out_dir", default=os.path.join(ROOT, "hasil"))
    ap.add_argument("--rocks", nargs="+", default=["Bentheimer", "BSG"])
    a = ap.parse_args()
    jalankan_komparasi(os.path.abspath(a.base_out_dir), a.rocks)


if __name__ == "__main__":
    main()
