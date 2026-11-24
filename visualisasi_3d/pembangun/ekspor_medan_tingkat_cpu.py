#!/usr/bin/env python3
"""
visualisasi-3d/ekspor_medan_tingkat_cpu.py
==========================================
Medan kecepatan LBM tingkat kasar (T1–T3, kedua cara, tiga arah) untuk penampil 3D, dihitung ulang di CPU.

Pemecah yang dibekukan tidak diubah: kernel Numba, pembangun domain, dan perjanjian satuannya diimpor dari
program/cpu/cpu_solver_lbm.py, lalu langkah waktunya diulang persis sampai langkah tunak yang tersimpan pada
hasil/gpu/<B>/simulasi_multiresolusi/<B>_<T>_<cara>_<arah>.json. Medan hanya disimpan bila k di langkah itu
sama dengan k tersimpan (selisih nisbi < 1e-9; paritas GPU–CPU 4e-12). Kasus tidak tembus dilewati.

Keluaran archive/medan-kecepatan/<B>_<T>_<cara>_<arah>.npz (indeks datar [z,y,x] pada grid tingkat n³ + ux, uy, uz)
dan .json pemeriksaannya. Jalankan dari kerja-s4:
    python visualisasi-3d/ekspor_medan_tingkat_cpu.py [--batuan ...] [--tingkat T1 T2 T3] [--cara ...] [--arah ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

KERJA = os.environ.get("KERJA_S4") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [KERJA, os.path.join(KERJA, "program", "cpu")]
import cpu_solver_lbm as S  # noqa: E402

HASIL = os.path.join(KERJA, "hasil", "gpu")
KELUAR = os.path.join(KERJA, "archive", "medan-kecepatan")
BATUAN = ["Bentheimer", "BSG", "BB", "BUG", "BanderaBrown", "BanderaGray", "Berea", "CastleGate", "Kirby", "Leopard", "Parker"]
ARAH_INT = {"x": 2, "y": 1, "z": 0}          # sumbu larik [z, y, x]
TAU, DRHO, PERIKSA, BATAS_K = 1.0, 0.002, 500, 1e-9


def jalankan(pori, arah, langkah, dx_um):
    s = ARAH_INT[arah]
    buffer_vox = max(1, int(round(9.0 / dx_um)))
    fluid, clean = S.bangun_domain_lbm(pori, arah_int=s, buffer_vox=buffer_vox)
    nx = clean.shape[0]
    roi = (slice(buffer_vox, buffer_vox + nx), slice(1, -1), slice(1, -1))
    planes = [buffer_vox - 1, buffer_vox + nx]
    c, w, opp = S.kisi_d3q19()
    nu, n = (TAU - 0.5) / 3.0, int(fluid.sum())
    idx = np.full(fluid.shape, -1, np.int32)
    idx[fluid] = np.arange(n, dtype=np.int32)
    nbr = []
    for ci in c:
        src = np.roll(idx, tuple(ci), axis=(0, 1, 2))
        if ci[0] > 0:
            src[:ci[0]] = -2
        if ci[0] < 0:
            src[ci[0]:] = -2
        nbr.append(src[fluid])
    nbr = np.ascontiguousarray(np.stack(nbr), dtype=np.int32)
    cf, wg, og = (np.ascontiguousarray(c, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64),
                  np.ascontiguousarray(opp, dtype=np.int32))
    rho_in, rho_out = 1.0 + DRHO / 2.0, 1.0 - DRHO / 2.0
    dens = np.broadcast_to(np.linspace(rho_in, rho_out, fluid.shape[0])[:, None, None], fluid.shape)[fluid]
    f = np.ascontiguousarray(wg[:, None] * dens[None, :], dtype=np.float64)
    fb = np.empty_like(f)
    faces = [np.ascontiguousarray(idx[t][fluid[t]], dtype=np.int32) for t in (0, -1)]
    roi_ids = np.ascontiguousarray(idx[roi][fluid[roi]], dtype=np.int32)
    plane_ids = [np.ascontiguousarray(idx[x][fluid[x]], dtype=np.int32) for x in planes]
    roi_size, L = int(np.prod(fluid[roi].shape)), abs(planes[1] - planes[0])
    zero, wp = np.zeros(3, dtype=np.float64), 1.0 / TAU
    S._pemanasan_jit(cf, wg, og)
    for _ in range(langkah):
        S._collision_numba(f, fb, cf, wg, og, zero, wp, wp, n)
        S._stream_numba(fb, f, nbr, og, n)
        S._pressure_numba(f, faces[0], cf, wg, og, 1, rho_in)
        S._pressure_numba(f, faces[1], cf, wg, og, -1, rho_out)
    sum_ux, rho0, rho1 = S._macro_roi_planes(f, cf, roi_ids, plane_ids[0], plane_ids[1])
    dp = float((rho0 - rho1) / 3.0)
    k = nu * float(sum_ux / roi_size) * L / dp
    rho = f[:, roi_ids].sum(0)
    u = (cf.T @ f[:, roi_ids]) / rho                       # komponen domain: (searah aliran, sumbu sisa 1, 2)
    pos = [a[roi_ids] for a in np.nonzero(fluid)] if False else None
    koord = np.nonzero(fluid)
    pos = [koord[i][roi_ids] for i in range(3)]
    sisa = [a for a in (0, 1, 2) if a != s]
    asli = [None, None, None]                              # koordinat pada sumbu larik asli [z, y, x]
    asli[s], asli[sisa[0]], asli[sisa[1]] = pos[0] - buffer_vox, pos[1] - 1, pos[2] - 1
    u_asli = [None, None, None]
    u_asli[s], u_asli[sisa[0]], u_asli[sisa[1]] = u[0], u[1], u[2]
    m = pori.shape[0]
    indeks = (asli[0].astype(np.int64) * m + asli[1]) * m + asli[2]
    urut = np.argsort(indeks)
    topeng = np.zeros(m ** 3, bool)
    topeng[indeks] = True
    geometri = bool(np.array_equal(topeng.reshape(m, m, m), np.moveaxis(clean, 0, s)))
    return k, indeks[urut].astype(np.int32), [u_asli[i][urut].astype(np.float32) for i in (2, 1, 0)], geometri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batuan", nargs="+", default=BATUAN)
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
                    keluar = os.path.join(KELUAR, f"{b}_{t}_{c}_{a}")
                    if os.path.exists(keluar + ".json"):
                        with open(keluar + ".json", encoding="utf-8") as h:
                            if json.load(h).get("cocok"):
                                continue
                    with open(os.path.join(HASIL, b, "simulasi_multiresolusi", f"{b}_{t}_{c}_{a}.json"), encoding="utf-8") as h:
                        acuan = json.load(h)
                    if not acuan.get("K_kisi") or not acuan.get("langkah"):
                        print(f"[{b} {t} {c} {a}] tidak tembus, dilewati", flush=True)
                        continue
                    t0 = time.monotonic()
                    k, indeks, (ux, uy, uz), geometri = jalankan(pori, a, int(acuan["langkah"]), float(acuan["dx_um"]))
                    galat = abs(k / acuan["K_kisi"] - 1)
                    cocok = bool(galat < BATAS_K and geometri)
                    info = {"batuan": b, "tingkat": t, "cara": c, "arah": a, "n": int(pori.shape[0]), "dx_um": acuan["dx_um"],
                            "urutan_indeks": "z,y,x", "satuan_kecepatan": "kisi", "langkah": int(acuan["langkah"]),
                            "K_kisi": k, "K_kisi_tersimpan": acuan["K_kisi"], "galat_nisbi_k": galat,
                            "geometri_cocok": geometri, "cocok": cocok, "durasi_detik": time.monotonic() - t0}
                    if cocok:
                        np.savez_compressed(keluar + ".npz", indeks=indeks, ux=ux, uy=uy, uz=uz)
                    with open(keluar + ".json", "w", encoding="utf-8") as h:
                        json.dump(info, h, indent=1)
                    print(f"[{b} {t} {c} {a}] {'COCOK' if cocok else 'TIDAK COCOK'} langkah={acuan['langkah']} "
                          f"galat_k={galat:.1e} {info['durasi_detik']:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
