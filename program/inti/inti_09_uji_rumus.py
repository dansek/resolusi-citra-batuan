#!/usr/bin/env python3
"""
program/inti/inti_09_uji_rumus.py
=================================
Uji rumus dan konvensi pipeline terhadap kasus berjawaban pasti (fase perbaikan, sebelum maraton).
Tidak memerlukan hasil simulasi; selesai dalam hitungan detik sampai satu menit.

  python program/inti/inti_09_uji_rumus.py            # GPU + CPU (bila CuPy tersedia)
  python program/inti/inti_09_uji_rumus.py --cpu      # hanya CPU

Yang diuji
  U1  konstanta Darcy dan konversi k kisi -> mD sama di semua modul
  U2  nilai_kelayakan (salinan P2) : galat_k = 2 * 0,5 / (d_t/dx / 2) dan batas putusan 2/4/8 voxel
  U3  konvensi sumbu C-order [z, y, x]: saluran searah x hanya menembus arah x, di SEMUA fungsi tembus
  U4  leher penentu (P16 `leher`): saluran sisi s -> r_c = s/2; leher sempit menentukan
  U5  Sv hitung muka (P11): kubus pori a^3 -> 6 a^2 muka; semua implementasi sama
  U6  penurunan resolusi: suara terbanyak (seri -> padat), rerata blok menjaga phi; GPU = CPU
  U7  pemecah LBM: GPU = CPU pada langkah tetap; saluran lurus tau_u = 1; arah tak tembus k = 0;
      (verifikasi analitik terhadap solusi eksak ada di Langkah 00)
  U8  pemisahan hasil uji/produksi pada hasil_sah
"""

import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from program.pemuat_modul import muat

GAGAL = []


def cek(nama, syarat, rinci=""):
    print(f"  [{'LULUS' if syarat else 'GAGAL'}] {nama}" + (f"  ({rinci})" if rinci else ""), flush=True)
    if not syarat:
        GAGAL.append(nama)


def saluran(N, s, sumbu=2, leher=None):
    """Pori berbentuk saluran persegi sisi s di tengah, searah `sumbu` (C-order: x=2, y=1, z=0).
    leher=(sisi, panjang): bagian tengah menyempit."""
    pori = np.zeros((N, N, N), bool)
    a = (N - s) // 2
    sl = [slice(a, a + s)] * 3
    sl[sumbu] = slice(None)
    pori[tuple(sl)] = True
    if leher:
        sn, pj = leher
        b = (N - sn) // 2
        m = N // 2
        sl = [slice(a, a + s)] * 3
        sl[sumbu] = slice(m - pj // 2, m + pj // 2)
        pori[tuple(sl)] = False
        sl = [slice(b, b + sn)] * 3
        sl[sumbu] = slice(m - pj // 2, m + pj // 2)
        pori[tuple(sl)] = True
    return pori


def uji_darcy():
    print("\nU1 Konstanta Darcy dan konversi k")
    import re
    nilai = {}
    for akar, _, berkas in os.walk(os.path.join(ROOT, "program")):
        for b in berkas:
            if b.endswith(".py"):
                for m in re.finditer(r"^DARCY\w*\s*=\s*([0-9.eE+-]+)", open(os.path.join(akar, b)).read(), re.M):
                    nilai[b] = float(m.group(1))
    cek("semua modul memakai 1 D = 9,869233e-13 m^2", all(abs(v - 9.869233e-13) < 1e-20 for v in nilai.values()),
        f"{len(nilai)} modul")
    k_mD = 1.0 * (2.25e-6) ** 2 / 9.869233e-13 * 1000
    cek("k_kisi = 1 voxel^2 pada dx 2,25 um = 5129,6 mD", abs(k_mD - 5129.57) < 0.05, f"{k_mD:.2f} mD")


def uji_kelayakan():
    print("\nU2 nilai_kelayakan (P2 sel 30)")
    m = muat("program/inti/inti_06_evaluasi_kelayakan_resolusi.py")
    for nis, kata in [(10, "BAIK"), (8, "BAIK"), (6, "LAYAK"), (3, "CATATAN"), (1.9, "NYQUIST")]:
        gk_harap = 2 * 0.5 / (nis / 2)            # P2: galat_r = 0,5 / r_voxel, galat_k = 2 galat_r
        _, gr, gk, put = m.nilai_kelayakan(nis * 2.25, 2.25)
        cek(f"d_t/dx = {nis}: galat_k {100*gk_harap:.1f}%, putusan memuat '{kata}'",
            abs(gk - gk_harap) < 1e-12 and kata in put.upper(), put)


def uji_sumbu(gpu):
    print("\nU3 Konvensi sumbu C-order [z, y, x]")
    N = 24
    p = saluran(N, 6, sumbu=2)          # searah x
    m2 = muat("program/inti/inti_02_rentang_segmentasi.py")
    m4 = muat("program/inti/inti_04_volume_elementer.py")
    m3 = muat("program/inti/inti_03_leher_penentu.py")
    peta = {"x": 2, "y": 1, "z": 0}
    for nama, f in [("inti_02.porositas_tembus", lambda a: m2.porositas_tembus(p, peta[a])),
                    ("inti_04.porositas_tembus", lambda a: m4.porositas_tembus(p, peta[a]))]:
        hasil = {a: f(a) > 0 for a in "xyz"}
        cek(f"{nama}: hanya x menembus", hasil == {"x": True, "y": False, "z": False}, str(hasil))
    for be in (["cpu", "gpu"] if gpu else ["cpu"]):
        s = muat(f"program/{be}/{be}_solver_lbm.py")
        m5 = muat(f"program/{be}/{be}_05_multiresolusi_lbm.py")
        hasil = {a: bool(s.bersihkan_domain(p, peta[a]).any()) for a in "xyz"}
        cek(f"{be}_solver.bersihkan_domain: hanya x menembus", hasil == {"x": True, "y": False, "z": False}, str(hasil))
        if be == "gpu":
            import cupy as cp
            hasil = {a: bool(m5.golongkan_tembus_gpu(cp.asarray(p), sumbu=peta[a]).any()) for a in "xyz"}
        else:
            hasil = {a: bool(m5.golongkan_tembus_cpu(p, sumbu=peta[a]).any()) for a in "xyz"}
        cek(f"{be}_05.golongkan_tembus: hanya x menembus", hasil == {"x": True, "y": False, "z": False}, str(hasil))
    r = m3.leher_penentu(p, 1.0)
    cek("inti_03.leher_penentu: arah pembatas bukan x (y, z tak menembus)", r["arah_pembatas"] != "x", str(r["r_c_voxel"]))


def uji_leher():
    print("\nU4 Leher penentu (P16 `leher`)")
    m3 = muat("program/inti/inti_03_leher_penentu.py")
    for s in (4, 8, 12):
        r = m3.leher_penentu(saluran(40, s, sumbu=2), 2.25)
        cek(f"saluran sisi {s}: r_c(x) = {s/2} voxel, d_c = {s*2.25:.2f} um",
            r["r_c_voxel"]["x"] == s / 2 and abs(r["d_c_um"]["x"] - s * 2.25) < 1e-9, str(r["r_c_voxel"]["x"]))
    r = m3.leher_penentu(saluran(40, 12, sumbu=0, leher=(4, 6)), 2.25)
    cek("saluran sisi 12 dengan leher sisi 4 (searah z): r_c(z) = 2 (leher menentukan)",
        r["r_c_voxel"]["z"] == 2.0 and r["arah_pembatas"] in "xyz", str(r["r_c_voxel"]))


def uji_sv(gpu):
    print("\nU5 Luas spesifik Sv (hitung muka, P11)")
    N, a, dx = 40, 10, 2.25
    p = np.zeros((N, N, N), bool)
    p[15:25, 15:25, 15:25] = True
    harap = 6 * a * a / (N ** 3 * dx)
    m2 = muat("program/inti/inti_02_rentang_segmentasi.py")
    m4 = muat("program/inti/inti_04_volume_elementer.py")
    cek("inti_02.luas_spesifik_um", abs(m2.luas_spesifik_um(p, dx) - harap) < 1e-15, f"{m2.luas_spesifik_um(p, dx):.6e}")
    cek("inti_04.luas_spesifik_um", abs(m4.luas_spesifik_um(p) - harap) < 1e-15, f"{m4.luas_spesifik_um(p):.6e}")
    m5c = muat("program/cpu/cpu_05_multiresolusi_lbm.py")
    v = m5c.hitung_luas_spesifik_cpu(p, dx * 1e-6)
    cek("cpu_05.hitung_luas_spesifik_cpu", abs(v - harap) / harap < 1e-12, f"{v:.6e}")
    if gpu:
        import cupy as cp
        m5g = muat("program/gpu/gpu_05_multiresolusi_lbm.py")
        v = m5g.hitung_luas_spesifik_gpu(cp.asarray(p), dx * 1e-6)
        cek("gpu_05.hitung_luas_spesifik_gpu", abs(v - harap) / harap < 1e-12, f"{v:.6e}")


def uji_penurunan(gpu):
    print("\nU6 Penurunan resolusi")
    m5c = muat("program/cpu/cpu_05_multiresolusi_lbm.py")
    blok = np.zeros((2, 2, 2), bool)
    blok.flat[:4] = True                                    # tepat separuh pori
    cek("suara terbanyak: seri 4/8 -> padat", not m5c.turunkan_suara_terbanyak_cpu(blok, 2).any())
    blok.flat[4] = True
    cek("suara terbanyak: 5/8 pori -> pori", bool(m5c.turunkan_suara_terbanyak_cpu(blok, 2).all()))
    rng = np.random.default_rng(1)
    vol = rng.random((64, 64, 64)) < 0.3
    for f in (2, 4, 8):
        b, T = m5c.turunkan_rerata_blok_cpu(vol, f, float(vol.mean()))
        semua = [float((vol.reshape(64 // f, f, 64 // f, f, 64 // f, f).mean(axis=(1, 3, 5)) > (k + 0.5) / f ** 3).mean())
                 for k in range(f ** 3)]
        terbaik = min(abs(s - vol.mean()) for s in semua)
        cek(f"rerata blok f={f}: phi paling dekat phi T0 dari semua ambang diskret",
            abs(abs(b.mean() - vol.mean()) - terbaik) < 1e-12, f"phi {b.mean():.4f} vs {vol.mean():.4f}, T={T:.4f}")
    if gpu:
        import cupy as cp
        m5g = muat("program/gpu/gpu_05_multiresolusi_lbm.py")
        for f in (2, 4, 8):
            mg = cp.asnumpy(m5g.turunkan_suara_terbanyak_gpu(cp.asarray(vol), f))
            bg, Tg = m5g.turunkan_rerata_blok_gpu(cp.asarray(vol), f, float(vol.mean()))
            bc, Tc = m5c.turunkan_rerata_blok_cpu(vol, f, float(vol.mean()))
            cek(f"GPU = CPU f={f} (suara terbanyak & rerata blok)",
                np.array_equal(mg, m5c.turunkan_suara_terbanyak_cpu(vol, f)) and np.array_equal(cp.asnumpy(bg), bc) and Tg == Tc)


def uji_lbm(gpu):
    print("\nU7 Pemecah LBM")
    p = saluran(24, 8, sumbu=2)
    hasil = {}
    for be in (["cpu", "gpu"] if gpu else ["cpu"]):
        s = muat(f"program/{be}/{be}_solver_lbm.py")
        hasil[be] = s.jalankan_kasus_lbm(p, "uji", 0, "uji", "x", maks_langkah=600, periksa=200, verbose=False)
        r0 = s.jalankan_kasus_lbm(p, "uji", 0, "uji", "y", maks_langkah=600, periksa=200, verbose=False)
        cek(f"{be}: arah y tak menembus -> k = 0 tanpa simulasi", r0["K_mD"] == 0.0 and r0["langkah"] == 0)
        tu = hasil[be]["diagnostik"].get("tau_u")
        # di mulut saluran aliran masih berkembang (reservoir 24^2 -> saluran 8^2), jadi tau_u sedikit > 1
        cek(f"{be}: saluran lurus tau_u ~ 1 (< 0,5%)", tu is not None and 1 <= tu < 1.005, f"{tu}")
        cek(f"{be}: kolom N tercatat = sisi citra (24)", hasil[be]["N"] == 24, str(hasil[be]["N"]))
    if gpu:
        kg, kc = hasil["gpu"]["K_kisi"], hasil["cpu"]["K_kisi"]
        cek("GPU = CPU pada 600 langkah (beda nisbi K < 1e-9)", abs(kg / kc - 1) < 1e-9, f"{abs(kg/kc-1):.2e}")
        rg = [r[2] for r in hasil["gpu"]["riwayat"]]
        rc = [r[2] for r in hasil["cpu"]["riwayat"]]
        # sisa ~1e-9 adalah selisih dua bilangan hampir sama: galat pembulatan teramplifikasi
        cek("GPU = CPU riwayat sisa (sampai pembulatan)", np.allclose(rg, rc, rtol=1e-3, atol=1e-12), f"{rg} vs {rc}")
    # verifikasi analitik (saluran persegi, pipa bundar, orde konvergensi) ada di Langkah 00.

def uji_pemisahan():
    print("\nU8 Pemisahan hasil uji / produksi")
    from program.katalog_batuan import hasil_sah, KRITERIA_KONVERGENSI
    d = {"kriteria_konvergensi": dict(KRITERIA_KONVERGENSI), "konvergen": True, "langkah": 10, "phi_th": 0.2}
    lama = os.environ.pop("S4_UJI", None)
    cek("produksi: berkas produksi sah", hasil_sah(d))
    cek("produksi: berkas uji ditolak", not hasil_sah(dict(d, mode="uji", sisi_uji=128)))
    os.environ["S4_UJI"] = "128"
    cek("uji: berkas produksi ditolak", not hasil_sah(d))
    cek("uji: berkas uji sisi sama sah", hasil_sah(dict(d, mode="uji", sisi_uji=128)))
    cek("uji: berkas uji sisi lain ditolak", not hasil_sah(dict(d, mode="uji", sisi_uji=64)))
    os.environ.pop("S4_UJI")
    if lama is not None:
        os.environ["S4_UJI"] = lama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="lewati uji GPU")
    a = ap.parse_args()
    gpu = False
    if not a.cpu:
        try:
            import cupy  # noqa: F401
            gpu = True
        except Exception:
            print("CuPy tidak tersedia; uji GPU dilewati.")
    os.environ.pop("S4_UJI", None)
    uji_darcy(); uji_kelayakan(); uji_sumbu(gpu); uji_leher(); uji_sv(gpu); uji_penurunan(gpu); uji_lbm(gpu); uji_pemisahan()
    print("\n" + ("SEMUA UJI LULUS" if not GAGAL else f"{len(GAGAL)} UJI GAGAL: " + "; ".join(GAGAL)))
    sys.exit(1 if GAGAL else 0)


if __name__ == "__main__":
    main()
