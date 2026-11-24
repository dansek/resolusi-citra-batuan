#!/usr/bin/env python3
"""
visualisasi-3d/bangun_data_panel.py
===================================
Data untuk penampil versi panel (web-panel/): Bentheimer dan BSG pada empat tingkat resolusi.

  * mesh pori per tingkat dan per cara penurunan resolusi, dari volume yang benar-benar disimulasikan
    (hasil/gpu/<B>/multiresolusi/<B>_<T>_<cara>.npy, True = pori),
  * streamline LBM resolusi penuh per arah (satu-satunya tingkat yang medan kecepatannya disimpan),
  * meta: φ, φ tembus, Sv, k per arah, k_geo, simpangan, d_t/Δx per tingkat dan cara (tabel hasil/gpu/<B>/),
    pita segmentasi, dan tingkat kritis pita + Nyquist.

Jalankan dari folder kerja-s4:  python visualisasi-3d/bangun_data_panel.py
Dari repo: KERJA_S4=<folder kerja-s4> KELUAR_PANEL=visualisasi_3d/data python visualisasi_3d/pembangun/bangun_data_panel.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bangun_data as B  # noqa: E402  (streamline, medan_padat, simpan, tulis_garis, tulis_mesh)

BATUAN = B.BATUAN
TINGKAT = [("T0", "resolusi_penuh", 2.25), ("T1", None, 4.5), ("T2", None, 9.0), ("T3", None, 18.0)]
CARA = ["majority", "block_avg"]
KELUAR = os.environ.get("KELUAR_PANEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "web-panel", "data")
MEDAN_TINGKAT = os.path.join(B.KERJA, "archive", "medan-kecepatan-tingkat")   # hasil ekspor_medan_tingkat_gpu.py
N_GARIS = 400


def mesh_tingkat(vol):
    """Isopermukaan pori dari volume biner tingkat mana pun, pada grid paling halus 64³ (koordinat voxel T0)."""
    n = vol.shape[0]
    f = max(1, n // 64)
    medan = vol.astype(np.float32)
    if f > 1:
        m = n // f
        medan = medan.reshape(m, f, m, f, m, f).mean(axis=(1, 3, 5))
    medan = ndi.gaussian_filter(medan, 0.6, mode="nearest")
    medan = np.pad(medan, 1, constant_values=0.0)
    v, fc, _, _ = marching_cubes(medan, 0.5)
    skala = B.N / (medan.shape[0] - 2)
    v = (v - 1.0 + 0.5) * skala                   # pusat sel grid -> koordinat voxel T0
    v = np.clip(v[:, ::-1], 0, B.N)               # (z, y, x) -> (x, y, z)
    return v.astype(np.float32), fc.astype(np.int64)


def angka(x):
    """Nilai tabel -> float, atau None bila kosong/NaN (kasus tidak tembus)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def garis_tingkat(b, t, c, n):
    """Streamline pada medan kecepatan tingkat kasar sendiri; titik dipetakan ke koordinat voxel T0."""
    f = B.N / n
    hasil = {}
    for a in "xyz":
        berkas = os.path.join(MEDAN_TINGKAT, f"{b}_{t}_{c}_{a}.npz")
        pemeriksaan = os.path.join(MEDAN_TINGKAT, f"{b}_{t}_{c}_{a}.json")
        if not os.path.exists(berkas):
            hasil[a] = None                      # tidak tembus, atau medan belum diekspor
            continue
        with open(pemeriksaan, encoding="utf-8") as h:
            assert json.load(h)["cocok"], (b, t, c, a)
        U = B.medan_tingkat(berkas, n)
        garis, laju, u_rerata, n_tembus, n_benih = B.streamline(U, a, N_GARIS, np.random.default_rng(2026))
        garis = [g * f + (f - 1) / 2.0 for g in garis]           # pusat sel tingkat -> koordinat voxel T0
        n_titik = B.tulis_garis(os.path.join(KELUAR, f"{b}_{t}_{c}_garis_{a}.js"), garis, laju, u_rerata) if garis else 0
        hasil[a] = {"garis": len(garis), "titik": n_titik, "tembus": n_tembus, "benih": n_benih} if garis else None
    return hasil


def baca_csv(*bagian):
    with open(os.path.join(*bagian), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(KELUAR, exist_ok=True)
    meta = {"N": B.N, "dx_um": 2.25, "skala_u16": B.SKALA_U16, "log_laju": [B.LOG_MIN, B.LOG_MAX],
            "urutan": BATUAN, "batuan": {}}
    for b in BATUAN:
        dir_b = os.path.join(B.HASIL, b)
        dua = {(r["tingkat"], r["cara"]): r for r in baca_csv(dir_b, "tabel_dua_cara_penurunan.csv")}
        simp = {(r["tingkat"], r["cara"]): r for r in baca_csv(dir_b, "tabel_simpangan_permeabilitas.csv")}
        kel = {r["tingkat"]: r for r in baca_csv(dir_b, "tabel_evaluasi_kelayakan_resolusi.csv")}
        kritis = {r["cara"]: r for r in B.baca_json(dir_b, "rekap_tingkat_kritis_fov.json")}
        info = B.info_batuan(b)
        info["pita_persen"] = angka(kritis["majority"]["setengah_lebar_segmentasi_persen"])
        info["kritis"] = {c: {"pita": kritis[c]["tingkat_kritis"], "dx_pita_um": kritis[c]["dx_kritis_um"],
                              "nyquist": kritis[c]["tingkat_kritis_nyquist"],
                              "dx_um": kritis[c]["dx_kritis_nyquist_um"],
                              "fov_volume": kritis[c]["lipatan_fov_nyquist_volume"],
                              "fov_arah": kritis[c]["lipatan_fov_nyquist_per_arah"]} for c in CARA}
        info["tingkat"] = {}
        for t, cara_t0, dx in TINGKAT:
            for c in ([cara_t0] if cara_t0 else CARA):
                vol = np.load(os.path.join(dir_b, "multiresolusi", f"{b}_{t}_{c}.npy"))
                r = dua[(t, c)]
                assert abs(vol.mean() - float(r["phi_tot"])) < 1e-6, (b, t, c, vol.mean(), r["phi_tot"])
                v, fc = mesh_tingkat(vol)
                kunci = f"{b}_{t}_{c}"
                B.tulis_mesh(os.path.join(KELUAR, f"{kunci}.js"), v, fc)
                s = simp.get((t, c), {})
                langkah, phi_th = {}, {}
                for a in "xyz":
                    berkas = os.path.join(dir_b, "simulasi_multiresolusi", f"{b}_{t}_{c}_{a}.json")
                    kasus = B.baca_json(berkas) if os.path.exists(berkas) else {}
                    langkah[a] = kasus.get("langkah")
                    phi_th[a] = kasus.get("phi_th")       # porositas tembus searah aliran a (tabel hanya memuat arah x)
                info["tingkat"][kunci] = {
                    "tingkat": t, "cara": c, "dx_um": dx, "n": vol.shape[0],
                    "phi_total": angka(r["phi_tot"]), "phi_tembus": angka(r["phi_th"]), "Sv_um1": angka(r["Sv_um1"]),
                    "pori_terputus_persen": angka(r["fraksi_pori_terputus_persen"]),
                    "k_mD": {"x": angka(r["kx_mD"]), "y": angka(r["ky_mD"]), "z": angka(r["kz_mD"])},
                    "k_geo_mD": angka(r["k_geo_mD"]),
                    "delta_persen": {"x": angka(s.get("delta_kx_persen", 0)), "y": angka(s.get("delta_ky_persen", 0)),
                                     "z": angka(s.get("delta_kz_persen", 0)), "geo": angka(s.get("delta_kgeo_persen", 0))},
                    "dt_per_dx": angka(kel[t]["dt_per_dx"]), "langkah": langkah, "phi_tembus_arah": phi_th,
                    "mesh_segitiga": int(len(fc))}
                print(f"  {kunci}: φ {float(r['phi_tot']):.4f}, {len(fc)} segitiga", flush=True)
                if t != "T0":
                    info["tingkat"][kunci]["arah"] = garis_tingkat(b, t, c, vol.shape[0])
        info["arah"] = {}
        for arah in "xyz":
            U, _ = B.medan_padat(b, arah)
            garis, laju, u_rerata, n_tembus, n_benih = B.streamline(U, arah, N_GARIS, np.random.default_rng(2026))
            n_titik = B.tulis_garis(os.path.join(KELUAR, f"{b}_garis_{arah}.js"), garis, laju, u_rerata)
            info["arah"][arah] = {"garis": len(garis), "titik": n_titik, "tembus": n_tembus, "benih": n_benih}
            del U
            print(f"  {b} {arah}: {len(garis)} garis", flush=True)
        info["tingkat"][f"{b}_T0_resolusi_penuh"]["arah"] = info["arah"]
        info.pop("mesh", None)
        meta["batuan"][b] = info
    with open(os.path.join(KELUAR, "meta.js"), "w", encoding="utf-8") as f:
        f.write("window.META_PANEL = " + json.dumps(meta, ensure_ascii=False) + ";\n")
    print("selesai")


if __name__ == "__main__":
    sys.exit(main())
