#!/usr/bin/env python3
"""
visualisasi-3d/ekspor_medan_tingkat_gpu.py  (dijalankan di server GPU, folder ~/resolusi-citra-batuan)
=========================================
Medan kecepatan LBM tingkat kasar (T1–T3, kedua cara, tiga arah) untuk penampil 3D.

Perulangan pemecah diambil dari archive/skrip/ekspor_medan_kecepatan.py (yang sudah dipakai untuk T0): kernel GPU
dan pembangun domain diimpor dari pemecah yang dibekukan, dan langkah diulang sampai langkah tunak tersimpan di
hasil/gpu/<B>/simulasi_multiresolusi/<B>_<T>_<cara>_<arah>.json. Medan disimpan hanya bila langkah sama, k sama
(selisih nisbi < 1e-9), dan simpul fluida sama dengan domain bersih. Kasus tidak tembus dilewati.
Pemetaan sumbu sama dengan ekspor_medan_tingkat_cpu.py (diuji cocok pada BSG T2 suara terbanyak x).

Keluaran archive/medan-kecepatan-tingkat/<B>_<T>_<cara>_<arah>.npz (indeks datar [z,y,x] pada grid n³ + ux, uy, uz)
    python visualisasi-3d/ekspor_medan_tingkat_gpu.py [--batuan ...] [--tingkat ...] [--cara ...] [--arah ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.expanduser("~/resolusi-citra-batuan")
sys.path[:0] = [os.path.join(ROOT, "archive", "skrip")]
import ekspor_medan_kecepatan as E  # noqa: E402  (simulasi, S = gpu_solver_lbm)

HASIL = os.path.join(ROOT, "hasil", "gpu")
KELUAR = os.path.join(ROOT, "archive", "medan-kecepatan-tingkat")
ARAH_INT = {"x": 2, "y": 1, "z": 0}
BATAS_K = 1e-9


def ekspor(b, t, c, a, pori):
    keluar = os.path.join(KELUAR, f"{b}_{t}_{c}_{a}")
    if os.path.exists(keluar + ".json"):
        with open(keluar + ".json", encoding="utf-8") as h:
            if json.load(h).get("cocok"):
                return
    with open(os.path.join(HASIL, b, "simulasi_multiresolusi", f"{b}_{t}_{c}_{a}.json"), encoding="utf-8") as h:
        acuan = json.load(h)
    if not acuan.get("K_kisi") or not acuan.get("langkah"):
        print(f"[{b} {t} {c} {a}] tidak tembus, dilewati", flush=True)
        return
    s = ARAH_INT[a]
    bv = max(1, int(round(9.0 / float(acuan["dx_um"]))))
    fluid, clean = E.S.bangun_domain_lbm(pori, arah_int=s, buffer_vox=bv)
    nx = clean.shape[0]
    roi = (slice(bv, bv + nx), slice(1, -1), slice(1, -1))
    t0 = time.monotonic()
    k, step, konvergen, dp, nu, L, u, pos = E.simulasi(fluid, roi, [bv - 1, bv + nx], int(acuan["langkah"]), f"{b}_{t}_{c}_{a}")
    sisa = [i for i in (0, 1, 2) if i != s]
    asli, ua = [None] * 3, [None] * 3
    asli[s], asli[sisa[0]], asli[sisa[1]] = pos[0] - bv, pos[1] - 1, pos[2] - 1
    ua[s], ua[sisa[0]], ua[sisa[1]] = u[0], u[1], u[2]
    m = pori.shape[0]
    indeks = (asli[0].astype(np.int64) * m + asli[1]) * m + asli[2]
    urut = np.argsort(indeks)
    topeng = np.zeros(m ** 3, bool)
    topeng[indeks] = True
    geometri = bool(np.array_equal(topeng.reshape(m, m, m), np.moveaxis(clean, 0, s)))
    galat = abs(k / acuan["K_kisi"] - 1)
    cocok = bool(step == int(acuan["langkah"]) and galat < BATAS_K and geometri)
    info = {"batuan": b, "tingkat": t, "cara": c, "arah": a, "n": int(m), "dx_um": acuan["dx_um"], "urutan_indeks": "z,y,x",
            "satuan_kecepatan": "kisi", "langkah": int(step), "langkah_tersimpan": int(acuan["langkah"]), "konvergen": bool(konvergen),
            "K_kisi": k, "K_kisi_tersimpan": acuan["K_kisi"], "galat_nisbi_k": galat, "geometri_cocok": geometri,
            "cocok": cocok, "durasi_detik": time.monotonic() - t0}
    if cocok:
        np.savez_compressed(keluar + ".npz", indeks=indeks[urut].astype(np.int32),
                            ux=ua[2][urut].astype(np.float32), uy=ua[1][urut].astype(np.float32), uz=ua[0][urut].astype(np.float32))
    with open(keluar + ".json", "w", encoding="utf-8") as h:
        json.dump(info, h, indent=1)
    print(f"[{b} {t} {c} {a}] {'COCOK' if cocok else 'TIDAK COCOK'} langkah={step}/{acuan['langkah']} galat_k={galat:.1e} "
          f"{info['durasi_detik']:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batuan", nargs="+", default=E.BATUAN)
    ap.add_argument("--tingkat", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--cara", nargs="+", default=["majority", "block_avg"])
    ap.add_argument("--arah", nargs="+", default=["x", "y", "z"])
    arg = ap.parse_args()
    os.makedirs(KELUAR, exist_ok=True)
    for b in arg.batuan:
        for t in arg.tingkat:
            for c in arg.cara:
                pori = np.load(os.path.join(HASIL, b, "multiresolusi", f"{b}_{t}_{c}.npy"))
                for a in arg.arah:
                    ekspor(b, t, c, a, pori)
    print("SELESAI", flush=True)


if __name__ == "__main__":
    sys.exit(main())
