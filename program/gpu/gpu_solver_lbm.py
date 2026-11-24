#!/usr/bin/env python3
"""
program/gpu/gpu_solver_lbm.py
=========================
Pemecah Aliran Lattice Boltzmann D3Q19 Bertekanan Zou-He Berakselerasi GPU (CuPy)
Proyek S4: Batas Kekasaran Citra (Sampai Seberapa Kasar Citra Masih Berguna)

Fisika dan Syarat Batas:
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
import numpy as np
from scipy import ndimage as ndi

try:
    import cupy as cp
except ImportError:
    print("[GALAT] Modul CuPy tidak ditemukan! Gunakan program/cpu/cpu_solver_lbm.py untuk mode CPU.")
    sys.exit(1)

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


COLLISION_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void collision(const double* f, double* fb,
 const double* c, const double* w, const int* opp, const double* F,
 double wp, double wm, int n, int Q, int D) {
 int j=blockDim.x*blockIdx.x+threadIdx.x; if(j>=n) return;
 double r=0, u[3]={0,0,0};
 for(int i=0;i<Q;i++) r+=f[(long long)i*n+j];
 for(int a=0;a<D;a++) {
  double m=0; for(int i=0;i<Q;i++) m+=c[i*D+a]*f[(long long)i*n+j];
  u[a]=(m+0.5*F[a])/r;
 }
 double uu=0,uF=0; for(int a=0;a<D;a++){uu+=u[a]*u[a];uF+=F[a]*u[a];}
 for(int i=0;i<Q;i++) {
  int o=opp[i]; if(o<i) continue;
  double cu=0,cF=0;for(int a=0;a<D;a++){cu+=c[i*D+a]*u[a];cF+=c[i*D+a]*F[a];}
  double ei=w[i]*r*(1+3*cu+4.5*cu*cu-1.5*uu);
  double eo=w[o]*r*(1-3*cu+4.5*cu*cu-1.5*uu);
  long long ij=(long long)i*n+j,oj=(long long)o*n+j;
  double fp=0.5*(f[ij]+f[oj]),fm=0.5*(f[ij]-f[oj]);
  double ep=0.5*(ei+eo),em=0.5*(ei-eo);
  double Sg=w[i]*(-3*uF+9*cu*cF),Sa=w[i]*3*cF;
  double gp=-wp*(fp-ep)+(1-0.5*wp)*Sg;
  double gm=-wm*(fm-em)+(1-0.5*wm)*Sa;
  fb[ij]=f[ij]+gp+gm;if(o!=i)fb[oj]=f[oj]+gp-gm;
 }
}
''', 'collision', options=('--fmad=false',))

STREAM_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void pull(const double* fb, double* f, const int* nbr, const int* opp, int n, int Q) {
 long long j=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(j>=(long long)n*Q) return;
 int i=j/n, p=j%n, s=nbr[j];
 f[j]=s>=0 ? fb[(long long)i*n+s] : (s==-1 ? fb[(long long)opp[i]*n+p] : 0.0);
}
''', 'pull')

PRESSURE_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void pressure(double* f, const int* ids, const double* c, const double* w,
 const int* opp, int count, int n, int sign, double rho) {
 int k=blockDim.x*blockIdx.x+threadIdx.x; if(k>=count) return; int j=ids[k];
 double known=0;
 for(int i=0;i<19;i++){
  int cx=(int)c[3*i];
  if(cx==0) known+=f[(long long)i*n+j];
  else if(cx==-sign) known+=2*f[(long long)i*n+j];
 }
 double su=1.0 - known/rho;
 for(int i=0;i<19;i++) if((int)c[3*i]==sign)
  f[(long long)i*n+j]=f[(long long)opp[i]*n+j] + 6.0*w[i]*rho*su;
 double my=0, mz=0;
 for(int i=0;i<19;i++){my+=c[3*i+1]*f[(long long)i*n+j]; mz+=c[3*i+2]*f[(long long)i*n+j];}
 for(int i=0;i<19;i++) if((int)c[3*i]==sign)
  f[(long long)i*n+j] -= 0.5*(c[3*i+1]*my + c[3*i+2]*mz);
}
''', 'pressure')


def bersihkan_domain(pori, sumbu=0):
    if isinstance(pori, cp.ndarray):
        pori = cp.asnumpy(pori)
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



def simulasikan_lbm(fluid, roi, pressure_planes, tau=1.0, drho=0.002, maks=30000, tol=1e-6, periksa=500, label="run", verbose=True, tol_fluks=1e-3):
    c, w, opp = kisi_d3q19()
    Q = 19
    nu = (tau - 0.5) / 3.0
    n = int(fluid.sum())
    if n == 0:
        return {"k": 0.0, "konvergen": True, "langkah": 0, "durasi_detik": 0.0, "riwayat": []}

    cp.get_default_memory_pool().free_all_blocks()   # puncak kolam di akhir = milik kasus ini saja
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
    nbr = cp.asarray(np.stack(nbr))
    cf = cp.asarray(c, dtype=cp.float64)
    wg = cp.asarray(w)
    og = cp.asarray(opp, dtype=cp.int32)

    rho_in = 1.0 + drho / 2.0
    rho_out = 1.0 - drho / 2.0
    density = np.broadcast_to(np.linspace(rho_in, rho_out, fluid.shape[0])[:, None, None], fluid.shape)[fluid]
    f = wg[:, None] * cp.asarray(density)[None, :]
    fb = cp.empty_like(f)

    faces = [cp.asarray(idx[side][fluid[side]], dtype=cp.int32) for side in [0, -1]]
    roi_ids = cp.asarray(idx[roi][fluid[roi]], dtype=cp.int32)
    plane_ids = [cp.asarray(idx[x][fluid[x]], dtype=cp.int32) for x in pressure_planes]
    roi_size = int(np.prod(fluid[roi].shape))
    L = abs(pressure_planes[1] - pressure_planes[0])
    zero = cp.zeros(3)

    # Indeks irisan (searah aliran) tiap simpul pori sampel, untuk keseragaman fluks antaririsan (P11)
    x_roi_np = (np.nonzero(fluid)[0][idx[roi][fluid[roi]]] - roi[0].start).astype(np.int64)
    n_irisan = fluid[roi].shape[0]
    luas_irisan = fluid[roi].shape[1] * fluid[roi].shape[2]
    x_roi = cp.asarray(x_roi_np)
    larik_gb = (f.nbytes + fb.nbytes + nbr.nbytes) / 1e9
    kolam = cp.get_default_memory_pool()
    from program import kondisi_perangkat
    snap_kondisi = kondisi_perangkat.mulai("gpu")
    history = []
    old_k = 0.0
    streak = 0
    converged = False
    cp.cuda.Stream.null.synchronize()
    start_time = time.monotonic()
    k = 0.0
    step = 0

    for step in range(1, maks + 1):
        COLLISION_KERNEL(((n + 127) // 128,), (128,), (f, fb, cf, wg, og, zero, cp.float64(1.0 / tau), cp.float64(1.0 / tau), np.int32(n), np.int32(Q), np.int32(3)))
        STREAM_KERNEL(((Q * n + 255) // 256,), (256,), (fb, f, nbr, og, np.int32(n), np.int32(Q)))
        for ids, sign, density_bc in zip(faces, [1, -1], [rho_in, rho_out]):
            PRESSURE_KERNEL(((len(ids) + 127) // 128,), (128,), (f, ids, cf, wg, og, np.int32(len(ids)), np.int32(n), np.int32(sign), cp.float64(density_bc)))

        if step % periksa == 0 or step == maks:
            rho = f.sum(0)
            u = (cf.T @ f) / rho
            q = float(u[0, roi_ids].sum() / roi_size)
            dp = float((rho[plane_ids[0]].mean() - rho[plane_ids[1]].mean()) / 3.0)
            k = nu * q * L / dp if dp > 0 else 0.0
            residual = abs(k - old_k) / max(abs(k), 1e-30)
            old_k = k
            fluks = cp.bincount(x_roi, weights=u[0, roi_ids], minlength=n_irisan) / luas_irisan
            sb_fluks = float(fluks.std() / fluks.mean()) if float(fluks.mean()) > 0 else float('inf')
            history.append((step, k, residual, q, dp, sb_fluks))
            streak = streak + 1 if (residual < tol and sb_fluks < tol_fluks) else 0
            if verbose:
                status_streak = f" [lolos {streak}/2]" if streak > 0 else ""
                print(f"    [{label} | GPU] iter={step:>6} | K_kisi={k:.6e} | res={residual:.3e} | fluks={sb_fluks:.2e}{status_streak} | ({time.monotonic()-start_time:.2f}s)", flush=True)
            if streak >= 2:   # P7/P11: sisa < tol DAN simpangan fluks antaririsan < tol_fluks, dua pemeriksaan berturut
                converged = True
                break


    cp.cuda.Stream.null.synchronize()
    durasi = time.monotonic() - start_time
    kondisi = kondisi_perangkat.selesai(snap_kondisi)
    mlups = (n * step) / (durasi * 1e6) if durasi > 0 else 0.0
    import resource
    memori = {"taksiran_P7_kisi_penuh_gb": int(np.prod(fluid.shape)) * 19 * 8 * 2 / 1e9,
              "larik_utama_gb": larik_gb, "kolam_gpu_puncak_gb": kolam.total_bytes() / 1e9,
              "rss_puncak_proses_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}

    # Diagnostik keabsahan aliran (P11 sifat_aliran / sifat_efektif), dihitung sekali di akhir:
    #  tau_u = sum|u| / sum u_x pada simpul pori sampel (tortuositas aliran), Ma = |u|_maks * sqrt(3),
    #  Re pori = <u_x>_pori * d_h / nu dengan d_h = 4 V_pori / A_muka, dan simpangan nisbi fluks antaririsan.
    rho_akhir = f.sum(0)
    u_akhir = (cf.T @ f) / rho_akhir
    u_roi = u_akhir[:, roi_ids]
    sum_ux = float(u_roi[0].sum())
    tau_u = float(cp.sqrt((u_roi ** 2).sum(0)).sum() / sum_ux) if sum_ux > 0 else float("nan")
    umaks = float(cp.sqrt((u_akhir ** 2).sum(0)).max())
    x_roi = np.nonzero(fluid)[0][cp.asnumpy(roi_ids)] - roi[0].start
    fluks_irisan = np.bincount(x_roi, weights=cp.asnumpy(u_roi[0]), minlength=fluid[roi].shape[0])
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
    """Fungsi pembungkus untuk menjalankan satu kasus simulasi LBM GPU dari array boolean pori."""
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
            print(f"    [{label} | GPU] Domain TIDAK MENEMBUS (phi_th = 0). K = 0.0 mD", flush=True)
        res = {
            "rock": rock, "tingkat": tingkat, "N": N, "dx_um": dx_um,
            "cara": cara, "arah": arah, "backend": "gpu",
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
        print(f"    [{label} | GPU] Mulai LBM D3Q19: domain={fluid.shape}, N={N}^3, fluida={int(fluid.sum()):,} vox, maks={maks_langkah}...", flush=True)

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
        print(f"    [{label} | GPU] Selesai: K_mD={k_mD:.2f} mD | Langkah={sim_res['langkah']} | Konvergen={status_konv} | {sim_res['durasi_detik']:.2f}s ({sim_res['mlups']:.1f} MLUPS)", flush=True)

    out_data = {
        "rock": rock, "tingkat": tingkat, "N": N, "dx_um": dx_um,
        "cara": cara, "arah": arah, "backend": "gpu",
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
    res = simulasikan_lbm(
        fluid, roi, pressure_planes,
        tau=1.0, drho=0.002, maks=100,
        tol=1e-12, periksa=100, label=f"BENCH_{rock}_{arah}", verbose=False
    )
    return res["durasi_detik"], res["mlups"]


def main():
    parser = argparse.ArgumentParser(description="Solver LBM D3Q19 GPU (CuPy)")
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
