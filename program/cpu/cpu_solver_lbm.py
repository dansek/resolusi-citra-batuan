#!/usr/bin/env python3
"""
program/cpu/cpu_solver_lbm.py
=========================
Pemecah Aliran Lattice Boltzmann D3Q19 Bertekanan Zou-He Berakselerasi CPU Paralel (Numba)
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Fisika dan Syarat Batas (100% Identik dengan Versi GPU):
- Operator Tumbukan: BGK (tau = 1.0, float64), nu_kisi = 1/6
- Penggerak Aliran: Beda kerapatan/tekanan Zou-He (drho = 0.002, dP_kisi = drho / 3)
- Syarat Batas Dinding: Bounce-back setengah jalan (half-way bounce back) pada padatan
- Lapisan Penyangga: Reservoir fluida di inlet dan outlet dengan tebal fisis tetap (~9 um)
  * N = 256 (dx = 2.25 um): 4 voxel
  * N = 128 (dx = 4.50 um): 2 voxel
  * N = 64  (dx = 9.00 um): 1 voxel
  * N = 32  (dx = 18.0 um): 1 voxel
- Sisi Samping Melintang: Dinding padat tanpa slip (1 lapis solid)
- Konvensi Permeabilitas Darcy:
  q_kisi = Q_fluida / A_sampel
  K_kisi = (q_kisi * nu * L_sampel) / dP_kisi
  K_mD   = (K_kisi * dx_m^2 / 0.9869233e-12) * 1000.0
"""

import os
import sys
import time
import json
import math
import argparse

# Pastikan inisialisasi hardware dan afinitas core fisik murni dipasang SEBELUM numba diimpor
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from program.utilitas_hardware import inisialisasi_lingkungan_hardware, terapkan_afinitas_core_fisik
    inisialisasi_lingkungan_hardware()
except Exception:
    pass

import numpy as np
from scipy import ndimage as ndi
import numba as nb

try:
    terapkan_afinitas_core_fisik()
except Exception:
    pass

DARCY_M2 = 0.9869233e-12


def kisi_d3q19():
    c = [[0, 0, 0]]
    for a in range(3):
        for s in (1, -1):
            v = [0, 0, 0]
            v[a] = s
            c.append(v)
    for a in range(3):
        for b in range(a + 1, 3):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0, 0, 0]
                    v[a] = s1
                    v[b] = s2
                    c.append(v)
    c = np.array(c, dtype=np.int32)
    w = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=np.float64)
    opp = np.array([np.where((c == -ci).all(1))[0][0] for ci in c], dtype=np.int32)
    return c, w, opp


@nb.njit(parallel=True)
def _collision_numba(f, fb, c, w, opp, F, wp, wm, n):
    Q = 19
    D = 3
    for j in nb.prange(n):
        r = 0.0
        for i in range(Q):
            r += f[i, j]
        u0 = 0.0
        u1 = 0.0
        u2 = 0.0
        for i in range(Q):
            fi = f[i, j]
            u0 += c[i, 0] * fi
            u1 += c[i, 1] * fi
            u2 += c[i, 2] * fi
        u0 = (u0 + 0.5 * F[0]) / r
        u1 = (u1 + 0.5 * F[1]) / r
        u2 = (u2 + 0.5 * F[2]) / r

        uu = u0 * u0 + u1 * u1 + u2 * u2
        uF = F[0] * u0 + F[1] * u1 + F[2] * u2

        for i in range(Q):
            o = opp[i]
            if o < i:
                continue
            cu = c[i, 0] * u0 + c[i, 1] * u1 + c[i, 2] * u2
            cF = c[i, 0] * F[0] + c[i, 1] * F[1] + c[i, 2] * F[2]
            ei = w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
            eo = w[o] * r * (1.0 - 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
            fp = 0.5 * (f[i, j] + f[o, j])
            fm = 0.5 * (f[i, j] - f[o, j])
            ep = 0.5 * (ei + eo)
            em = 0.5 * (ei - eo)
            Sg = w[i] * (-3.0 * uF + 9.0 * cu * cF)
            Sa = w[i] * 3.0 * cF
            gp = -wp * (fp - ep) + (1.0 - 0.5 * wp) * Sg
            gm = -wm * (fm - em) + (1.0 - 0.5 * wm) * Sa
            fb[i, j] = f[i, j] + gp + gm
            if o != i:
                fb[o, j] = f[o, j] + gp - gm


@nb.njit(parallel=True)
def _stream_numba(fb, f, nbr, opp, n):
    Q = 19
    for i in range(Q):
        oi = opp[i]
        for p in nb.prange(n):
            s = nbr[i, p]
            if s >= 0:
                f[i, p] = fb[i, s]
            elif s == -1:
                f[i, p] = fb[oi, p]
            else:
                f[i, p] = 0.0


@nb.njit(parallel=True)
def _pressure_numba(f, ids, c, w, opp, sign, rho_bc):
    count = len(ids)
    for k in nb.prange(count):
        j = ids[k]
        known = 0.0
        for i in range(19):
            cx = int(c[i, 0])
            if cx == 0:
                known += f[i, j]
            elif cx == -sign:
                known += 2.0 * f[i, j]
        su = 1.0 - known / rho_bc
        for i in range(19):
            if int(c[i, 0]) == sign:
                f[i, j] = f[opp[i], j] + 6.0 * w[i] * rho_bc * su
        my = 0.0
        mz = 0.0
        for i in range(19):
            my += c[i, 1] * f[i, j]
            mz += c[i, 2] * f[i, j]
        for i in range(19):
            if int(c[i, 0]) == sign:
                f[i, j] -= 0.5 * (c[i, 1] * my + c[i, 2] * mz)


@nb.njit(parallel=True)
def _macro_roi_planes(f, c, roi_ids, plane0_ids, plane1_ids):
    n_roi = len(roi_ids)
    sum_ux = 0.0
    for k in nb.prange(n_roi):
        j = roi_ids[k]
        r = 0.0
        mx = 0.0
        for i in range(19):
            fi = f[i, j]
            r += fi
            mx += c[i, 0] * fi
        sum_ux += mx / r

    n_p0 = len(plane0_ids)
    sum_rho0 = 0.0
    for k in nb.prange(n_p0):
        j = plane0_ids[k]
        r = 0.0
        for i in range(19):
            r += f[i, j]
        sum_rho0 += r

    n_p1 = len(plane1_ids)
    sum_rho1 = 0.0
    for k in nb.prange(n_p1):
        j = plane1_ids[k]
        r = 0.0
        for i in range(19):
            r += f[i, j]
        sum_rho1 += r

    return sum_ux, sum_rho0 / n_p0, sum_rho1 / n_p1


@nb.njit(cache=False)
def _fluks_irisan_numba(f, c, roi_ids, x_roi, n_irisan):
    """Jumlah u_x per irisan tegak lurus aliran pada simpul pori sampel (keseragaman fluks, P11)."""
    hasil = np.zeros(n_irisan)
    for k in range(len(roi_ids)):
        j = roi_ids[k]
        r = 0.0
        mx = 0.0
        for i in range(19):
            fi = f[i, j]
            r += fi
            mx += c[i, 0] * fi
        hasil[x_roi[k]] += mx / r
    return hasil


def bersihkan_domain(pori, sumbu=0):
    v = np.ascontiguousarray(np.moveaxis(pori, sumbu, 0))
    lbl, count = ndi.label(v, structure=ndi.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros_like(v, dtype=bool)
    inlet = set(np.unique(lbl[0])) - {0}
    outlet = set(np.unique(lbl[-1])) - {0}
    conn = inlet & outlet
    if not conn:
        return np.zeros_like(v, dtype=bool)
    return np.isin(lbl, list(conn))


def bangun_domain_lbm(pori, arah_int=0, buffer_vox=4):
    clean = bersihkan_domain(pori, arah_int)
    nx, ny, nz = clean.shape
    fluid = np.zeros((nx + 2 * buffer_vox, ny + 2, nz + 2), dtype=bool)
    fluid[:buffer_vox, 1:-1, 1:-1] = True
    fluid[-buffer_vox:, 1:-1, 1:-1] = True
    fluid[buffer_vox:-buffer_vox, 1:-1, 1:-1] = clean
    return fluid, clean



_SUDAH_PANAS = False


def _pemanasan_jit(cf, wg, og):
    """Kompilasi sekali semua kernel Numba pada domain 2 simpul, agar durasi run tidak memuat waktu kompilasi."""
    global _SUDAH_PANAS
    if _SUDAH_PANAS:
        return
    n = 2
    f = np.ascontiguousarray(wg[:, None] * np.ones((1, n)))
    fb = np.empty_like(f)
    nbr = np.full((19, n), -1, dtype=np.int32)
    ids = np.arange(n, dtype=np.int32)
    _collision_numba(f, fb, cf, wg, og, np.zeros(3), 1.0, 1.0, n)
    _stream_numba(fb, f, nbr, og, n)
    _pressure_numba(f, ids, cf, wg, og, 1, 1.0)
    _macro_roi_planes(f, cf, ids, ids, ids)
    _fluks_irisan_numba(f, cf, ids, np.zeros(n, dtype=np.int64), 1)
    _SUDAH_PANAS = True


def simulasikan_lbm(fluid, roi, pressure_planes, tau=1.0, drho=0.002, maks=30000, tol=1e-6, periksa=500, label="run", verbose=True, tol_fluks=1e-3):
    c, w, opp = kisi_d3q19()
    nu = (tau - 0.5) / 3.0
    n = int(fluid.sum())
    if n == 0:
        return {"k": 0.0, "konvergen": True, "langkah": 0, "durasi_detik": 0.0, "riwayat": []}

    idx = np.full(fluid.shape, -1, np.int32)
    idx[fluid] = np.arange(n, dtype=np.int32)
    nbr = []
    for ci in c:
        source = np.roll(idx, tuple(ci), axis=(0, 1, 2))
        if ci[0] > 0:
            source[:ci[0]] = -2
        if ci[0] < 0:
            source[ci[0]:] = -2
        nbr.append(source[fluid])
    nbr = np.ascontiguousarray(np.stack(nbr), dtype=np.int32)
    cf = np.ascontiguousarray(c, dtype=np.float64)
    wg = np.ascontiguousarray(w, dtype=np.float64)
    og = np.ascontiguousarray(opp, dtype=np.int32)

    rho_in = 1.0 + drho / 2.0
    rho_out = 1.0 - drho / 2.0
    density = np.broadcast_to(np.linspace(rho_in, rho_out, fluid.shape[0])[:, None, None], fluid.shape)[fluid]
    f = np.ascontiguousarray(wg[:, None] * density[None, :], dtype=np.float64)
    fb = np.empty_like(f)

    faces = [np.ascontiguousarray(idx[side][fluid[side]], dtype=np.int32) for side in [0, -1]]
    roi_ids = np.ascontiguousarray(idx[roi][fluid[roi]], dtype=np.int32)
    plane_ids = [np.ascontiguousarray(idx[x][fluid[x]], dtype=np.int32) for x in pressure_planes]
    roi_size = int(np.prod(fluid[roi].shape))
    L = abs(pressure_planes[1] - pressure_planes[0])
    zero = np.zeros(3, dtype=np.float64)
    wp = wm = 1.0 / tau

    # Pemanasan JIT Numba: kompilasi kernel pada larik kecil di luar pengukuran waktu
    _pemanasan_jit(cf, wg, og)
    # Indeks irisan (searah aliran) tiap simpul pori sampel, untuk keseragaman fluks antaririsan (P11)
    x_roi_np = (np.nonzero(fluid)[0][idx[roi][fluid[roi]]] - roi[0].start).astype(np.int64)
    n_irisan = fluid[roi].shape[0]
    luas_irisan = fluid[roi].shape[1] * fluid[roi].shape[2]
    larik_gb = (f.nbytes + fb.nbytes + nbr.nbytes) / 1e9
    from program import kondisi_perangkat
    snap_kondisi = kondisi_perangkat.mulai("cpu")
    history = []
    old_k = 0.0
    streak = 0
    converged = False
    start_time = time.monotonic()
    k = 0.0
    step = 0

    for step in range(1, maks + 1):
        _collision_numba(f, fb, cf, wg, og, zero, wp, wm, n)
        _stream_numba(fb, f, nbr, og, n)
        _pressure_numba(f, faces[0], cf, wg, og, 1, rho_in)
        _pressure_numba(f, faces[1], cf, wg, og, -1, rho_out)

        if step % periksa == 0 or step == maks:
            sum_ux, rho0, rho1 = _macro_roi_planes(f, cf, roi_ids, plane_ids[0], plane_ids[1])
            q = float(sum_ux / roi_size)
            dp = float((rho0 - rho1) / 3.0)
            k = nu * q * L / dp if dp > 0 else 0.0
            residual = abs(k - old_k) / max(abs(k), 1e-30)
            old_k = k
            fluks = _fluks_irisan_numba(f, cf, roi_ids, x_roi_np, n_irisan) / luas_irisan
            sb_fluks = float(fluks.std() / fluks.mean()) if fluks.mean() > 0 else float('inf')
            history.append((step, k, residual, q, dp, sb_fluks))
            streak = streak + 1 if (residual < tol and sb_fluks < tol_fluks) else 0
            if verbose:
                status_streak = f" [lolos {streak}/2]" if streak > 0 else ""
                print(f"    [{label} | CPU] iter={step:>6} | K_kisi={k:.6e} | res={residual:.3e} | fluks={sb_fluks:.2e}{status_streak} | ({time.monotonic()-start_time:.2f}s)", flush=True)
            if streak >= 2:   # P7/P11: sisa < tol DAN simpangan fluks antaririsan < tol_fluks, dua pemeriksaan berturut
                converged = True
                break


    durasi = time.monotonic() - start_time
    kondisi = kondisi_perangkat.selesai(snap_kondisi)
    mlups = (n * step) / (durasi * 1e6) if durasi > 0 else 0.0
    import resource
    memori = {"taksiran_P7_kisi_penuh_gb": int(np.prod(fluid.shape)) * 19 * 8 * 2 / 1e9,
              "larik_utama_gb": larik_gb,
              "rss_puncak_proses_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}

    # Diagnostik keabsahan aliran (P11 sifat_aliran / sifat_efektif), dihitung sekali di akhir:
    #  tau_u = sum|u| / sum u_x pada simpul pori sampel (tortuositas aliran), Ma = |u|_maks * sqrt(3),
    #  Re pori = <u_x>_pori * d_h / nu dengan d_h = 4 V_pori / A_muka, dan simpangan nisbi fluks antaririsan.
    rho_akhir = f.sum(0)
    u_akhir = (cf.T @ f) / rho_akhir
    u_roi = u_akhir[:, roi_ids]
    sum_ux = float(u_roi[0].sum())
    tau_u = float(np.sqrt((u_roi ** 2).sum(0)).sum() / sum_ux) if sum_ux > 0 else float("nan")
    umaks = float(np.sqrt((u_akhir ** 2).sum(0)).max())
    x_roi = np.nonzero(fluid)[0][np.asarray(roi_ids)] - roi[0].start
    fluks_irisan = np.bincount(x_roi, weights=np.asarray(u_roi[0]), minlength=fluid[roi].shape[0])
    fluks_irisan = fluks_irisan / (fluid[roi].shape[1] * fluid[roi].shape[2])
    sb_fluks = float(fluks_irisan.std() / fluks_irisan.mean()) if fluks_irisan.mean() > 0 else float("nan")
    pori_roi = fluid[roi]
    muka = sum(int(np.sum(pori_roi != np.roll(pori_roi, 1, axis=a))) for a in range(3))
    d_h = 4.0 * int(pori_roi.sum()) / muka if muka > 0 else 0.0
    u_pori = sum_ux / max(int(pori_roi.sum()), 1)
    diagnostik = {
        "tau_u": tau_u,
        "Ma": umaks * math.sqrt(3.0),
        "Re_pori": u_pori * d_h / nu,
        "d_h_voxel": d_h,
        "simpangan_fluks_irisan": sb_fluks,
    }

    return {
        "k": k,
        "konvergen": converged,
        "langkah": step,
        "durasi_detik": durasi,
        "mlups": mlups,
        "n_fluida": n,
        "riwayat": history,
        "diagnostik": diagnostik,
        "kriteria_konvergensi": {"tol": tol, "tol_fluks": tol_fluks, "periksa": periksa, "berturut": 2},
        "memori": memori,
        "kondisi_perangkat": kondisi
    }


def jalankan_kasus_lbm(pori, rock, tingkat, cara, arah, out_file=None, maks_langkah=None, periksa=500, tol=1e-6, verbose=True, tol_fluks=1e-3):
    """Fungsi pembungkus untuk menjalankan satu kasus simulasi LBM CPU dari array boolean pori."""
    arah_map = {"x": 2, "y": 1, "z": 0}  # C-order array [z, y, x]: z=0, y=1, x=2
    arah_idx = arah_map[arah]
    N = int(max(pori.shape))          # sisi citra sebenarnya (subvolume REV / potongan uji ikut tercatat benar)
    dx_um = 2.25 * (2 ** tingkat)
    dx_m = dx_um * 1e-6

    buffer_vox = max(1, int(round(9.0 / dx_um)))
    fluid, clean = bangun_domain_lbm(pori, arah_int=arah_idx, buffer_vox=buffer_vox)
    phi_th = float(clean.mean())

    label = f"{rock}_T{tingkat}_{cara}_{arah}"
    if phi_th == 0.0:
        if verbose:
            print(f"    [{label} | CPU] Domain TIDAK MENEMBUS (phi_th = 0). K = 0.0 mD", flush=True)
        res = {
            "rock": rock, "tingkat": tingkat, "N": N, "dx_um": dx_um,
            "cara": cara, "arah": arah, "backend": "cpu",
            "phi_tot": float(pori.mean()), "phi_th": 0.0,
            "K_kisi": 0.0, "K_m2": 0.0, "K_mD": 0.0,
            "konvergen": True, "langkah": 0, "durasi_detik": 0.0,
            "mlups": 0.0, "riwayat": []
        }
        from program import mode_uji
        res.update(mode_uji.tanda())
        if out_file:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        return res

    nx = clean.shape[0]
    roi = (slice(buffer_vox, buffer_vox + nx), slice(1, -1), slice(1, -1))
    pressure_planes = [buffer_vox - 1, buffer_vox + nx]

    if maks_langkah is None:
        try:
            from program.katalog_batuan import muat_semua_batuan, dapatkan_batas_langkah_batuan
            kat = muat_semua_batuan()
            maks_langkah = dapatkan_batas_langkah_batuan(kat, rock, tingkat=tingkat, arah=arah)
        except Exception:
            maks_langkah = 500000 if (rock == "Parker" and tingkat == 0) else (15000 if tingkat == 0 else 6000)

    from program import mode_uji
    maks_langkah, periksa = mode_uji.atur_lbm(maks_langkah, periksa)

    if verbose:
        print(f"    [{label} | CPU] Mulai LBM D3Q19: domain={fluid.shape}, N={N}^3, fluida={int(fluid.sum()):,} vox, maks={maks_langkah}...", flush=True)

    sim_res = simulasikan_lbm(
        fluid, roi, pressure_planes,
        tau=1.0, drho=0.002, maks=maks_langkah,
        tol=tol, periksa=periksa, label=label, verbose=verbose, tol_fluks=tol_fluks
    )

    k_kisi = sim_res["k"]
    k_m2 = k_kisi * (dx_m ** 2)
    k_mD = (k_m2 / DARCY_M2) * 1000.0

    if verbose:
        status_konv = "TUNAK (True)" if sim_res['konvergen'] else "Belum Tunak (False)"
        print(f"    [{label} | CPU] Selesai: K_mD={k_mD:.2f} mD | Langkah={sim_res['langkah']} | Konvergen={status_konv} | {sim_res['durasi_detik']:.2f}s ({sim_res['mlups']:.1f} MLUPS)", flush=True)

    out_data = {
        "rock": rock, "tingkat": tingkat, "N": N, "dx_um": dx_um,
        "cara": cara, "arah": arah, "backend": "cpu",
        "phi_tot": float(pori.mean()), "phi_th": phi_th,
        "K_kisi": k_kisi, "K_m2": k_m2, "K_mD": k_mD,
        "konvergen": sim_res["konvergen"], "langkah": sim_res["langkah"],
        "durasi_detik": sim_res["durasi_detik"], "mlups": sim_res["mlups"],
        "riwayat": sim_res["riwayat"],
        "diagnostik": sim_res.get("diagnostik", {}),
        "kriteria_konvergensi": sim_res.get("kriteria_konvergensi", {}),
        "n_fluida": sim_res.get("n_fluida", 0),
        "memori": sim_res.get("memori", {}),
        "kondisi_perangkat": sim_res.get("kondisi_perangkat", {})
    }
    from program import mode_uji
    out_data.update(mode_uji.tanda())
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
    return out_data


def benchmark_100_langkah(pori, rock="Bentheimer", arah="x"):
    """Menjalankan 100 langkah benchmark pada resolusi penuh 256^3 untuk estimasi waktu (ETA)."""
    arah_map = {"x": 2, "y": 1, "z": 0}  # C-order array [z, y, x]: z=0, y=1, x=2
    fluid, clean = bangun_domain_lbm(pori, arah_int=arah_map[arah], buffer_vox=4)
    nx = clean.shape[0]
    roi = (slice(4, 4 + nx), slice(1, -1), slice(1, -1))
    pressure_planes = [3, 4 + nx]
    # Warm-up JIT compilation (2 langkah) agar waktu kompilasi tidak masuk hitungan benchmark
    simulasikan_lbm(
        fluid, roi, pressure_planes,
        tau=1.0, drho=0.002, maks=2,
        tol=1e-12, periksa=2, label="WARMUP", verbose=False
    )
    res = simulasikan_lbm(
        fluid, roi, pressure_planes,
        tau=1.0, drho=0.002, maks=100,
        tol=1e-12, periksa=100, label=f"BENCH_{rock}_{arah}", verbose=False
    )
    return res["durasi_detik"], res["mlups"]


def main():
    parser = argparse.ArgumentParser(description="Solver LBM D3Q19 CPU (Numba Paralel)")
    parser.add_argument("--rock", required=True, help="Nama/kunci sampel batuan pasir")
    parser.add_argument("--tingkat", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--cara", required=True, choices=["resolusi_penuh", "majority", "block_avg"])
    parser.add_argument("--arah", required=True, choices=["x", "y", "z"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="hasil/simulasi_multiresolusi")
    parser.add_argument("--maks_langkah", type=int, default=None)
    parser.add_argument("--periksa", type=int, default=500)
    args = parser.parse_args()

    N = 256 // (2 ** args.tingkat)
    if args.tingkat == 0:
        raw_name = f"{args.rock}_2d25um_binary_256.raw"
        kandidat = [
            os.path.join(args.data_dir, "sumber", "citra-biner", raw_name),
            os.path.join(args.data_dir, "citra-biner", raw_name),
            os.path.join(args.data_dir, raw_name),
        ]
        p_raw = next((k for k in kandidat if os.path.exists(k)), None)
        if not p_raw:
            raise FileNotFoundError(f"Berkas {raw_name} tidak ditemukan di {args.data_dir}")
        v = np.fromfile(p_raw, dtype=np.uint8)
        pori = (v.reshape((256, 256, 256)) == 0)
    else:
        npy_name = f"{args.rock}_{args.cara}_{N}.npy"
        kandidat = [
            os.path.join(args.out_dir, "..", "multiresolusi", npy_name),
            os.path.join(args.data_dir, "multiresolusi", npy_name),
            os.path.join(args.data_dir, npy_name),
        ]
        p_npy = next((k for k in kandidat if os.path.exists(k)), None)
        if not p_npy:
            raise FileNotFoundError(f"Berkas {npy_name} tidak ditemukan!")
        pori = np.load(p_npy)

    t_label = f"T{args.tingkat}"
    out_file = os.path.join(args.out_dir, f"{args.rock}_{t_label}_{args.cara}_{args.arah}.json")
    jalankan_kasus_lbm(
        pori, args.rock, args.tingkat, args.cara, args.arah,
        out_file=out_file, maks_langkah=args.maks_langkah, periksa=args.periksa
    )


if __name__ == "__main__":
    main()
