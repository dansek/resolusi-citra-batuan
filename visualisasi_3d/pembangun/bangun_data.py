#!/usr/bin/env python3
"""
visualisasi-3d/bangun_data.py
=============================
Membangun data penampil 3D sebelas batupasir dari hasil maraton (bukan dari angka lama):

  * mesh permukaan pori per batuan, dari citra biner 256³ (rerata blok 4³ lalu marching cubes pada φ = 0,5),
  * streamline LBM per batuan dan per arah, diintegrasikan pada medan kecepatan resolusi penuh
    archive/medan-kecepatan/<batuan>_T0_<arah>.npz (indeks datar [z,y,x] simpul pori + ux, uy, uz satuan kisi),
  * meta.json: besaran resolusi penuh dari hasil/gpu/<batuan>/ (φ, k per arah, k_geo, d_t) dan,
    hanya untuk Bentheimer dan BSG, tingkat kritis yang dibahas laporan.

Keluaran ke web/data/. Jalankan dari folder kerja-s4:
    python visualisasi-3d/bangun_data.py [--batuan Bentheimer BSG] [--arah x y z] [--garis 400]
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

N = 256
KERJA = os.environ.get("KERJA_S4") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # folder kerja dengan hasil/gpu 11 batuan dan archive/medan-kecepatan
CITRA = os.path.join(KERJA, "data", "sumber", "citra-biner")
MEDAN = os.path.join(KERJA, "archive", "medan-kecepatan")
HASIL = os.path.join(KERJA, "hasil", "gpu")
KELUAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "data")

BATUAN = ["Bentheimer", "BSG", "BB", "BUG", "BanderaBrown", "BanderaGray", "Berea",
          "CastleGate", "Kirby", "Leopard", "Parker"]
LAPORAN = {"Bentheimer", "BSG"}
SUMBU = {"x": 0, "y": 1, "z": 2}          # urutan koordinat titik: (x, y, z)
SKALA_U16 = 65535.0 / N                  # koordinat voxel 0..256 -> uint16
LOG_MIN, LOG_MAX = -2.0, 1.5             # log10(|u|/ū) dipetakan ke 0..255


def baca_json(*bagian):
    with open(os.path.join(*bagian), encoding="utf-8") as f:
        return json.load(f)


def baca_citra(b):
    return np.fromfile(os.path.join(CITRA, f"{b}_2d25um_binary_256.raw"), np.uint8).reshape(N, N, N)


def mesh_pori(img, f=4):
    """Permukaan pori pada grid 64³: rerata blok f³ (fraksi pori), dihaluskan ringan, isopermukaan 0,5."""
    pori = (img == 0).astype(np.float32)
    m = N // f
    blok = pori.reshape(m, f, m, f, m, f).mean(axis=(1, 3, 5))
    blok = ndi.gaussian_filter(blok, 0.6, mode="nearest")
    # bantalan padatan agar permukaan tertutup di tepi domain
    blok = np.pad(blok, 1, constant_values=0.0)
    v, fc, _, _ = marching_cubes(blok, 0.5)
    v = (v - 1.0) * f + (f - 1) / 2.0     # grid blok -> koordinat voxel, urutan (z, y, x)
    v = np.clip(v[:, ::-1], 0, N)         # -> (x, y, z)
    return v.astype(np.float32), fc.astype(np.int64)


def medan_tingkat(berkas, n):
    """Medan kecepatan padat U[k, z, y, x] dari berkas npz (indeks datar [z,y,x] pada grid n³)."""
    d = np.load(berkas)
    U = np.zeros((3, n * n * n), np.float32)
    for k, nama in enumerate(("ux", "uy", "uz")):
        U[k, d["indeks"]] = d[nama]
    return U.reshape(3, n, n, n)


def medan_padat(b, arah):
    d = np.load(os.path.join(MEDAN, f"{b}_T0_{arah}.npz"))
    U = np.zeros((3, N * N * N), np.float32)
    for k, nama in enumerate(("ux", "uy", "uz")):
        U[k, d["indeks"]] = d[nama]
    return U.reshape(3, N, N, N), len(d["indeks"])  # U[k, z, y, x]


def interp(U, p):
    """Kecepatan trilinear di titik p (n, 3) berurutan (x, y, z); di luar pori bernilai nol."""
    c = np.stack([p[:, 2], p[:, 1], p[:, 0]])  # (z, y, x)
    return np.stack([ndi.map_coordinates(U[k], c, order=1, mode="constant", cval=0.0) for k in range(3)], axis=1)


def streamline(U, arah, n_garis, rng, h=0.5, maks_langkah=4000, simpan_tiap=3):
    n = U.shape[1]                              # sisi grid (256 pada T0, 128/64/32 pada T1–T3)
    a = SUMBU[arah]
    ua = U[a]                                   # komponen searah aliran, [z, y, x]
    besar = np.sqrt((U ** 2).sum(axis=0))
    u_rerata = float(besar[besar > 0].mean())   # rerata |u| atas simpul fluida, skala warna
    del besar
    # benih: simpul pori pada irisan masuk (indeks 3), dipilih sebanding fluks searah aliran
    q = np.take(ua, 3, axis=2 - a)              # sumbu array: z=0, y=1, x=2
    q = np.where(q > 0, q, 0)
    lokasi = np.argwhere(q > 0)
    bobot = q[q > 0].astype(np.float64)
    pilih = rng.choice(len(lokasi), size=min(n_garis * 3, len(lokasi)), replace=False, p=bobot / bobot.sum())
    benih = np.zeros((len(pilih), 3), np.float64)
    sumbu_lain = [s for s in range(3) if s != a]  # koordinat titik (x,y,z) selain arah aliran
    # lokasi adalah indeks pada larik 2D hasil np.take, berurutan menurut sumbu array tersisa (menaik)
    sumbu_array_lain = [s for s in (0, 1, 2) if s != 2 - a]       # sumbu array (z=0,y=1,x=2)
    for kolom, sa in enumerate(sumbu_array_lain):
        benih[:, 2 - sa] = lokasi[pilih, kolom] + rng.uniform(-0.3, 0.3, len(pilih))
    benih[:, a] = 3.0
    p = benih.copy()
    hidup = np.ones(len(p), bool)
    jejak = [[p[i].copy()] for i in range(len(p))]
    laju = [[float(np.linalg.norm(interp(U, p[i:i + 1])))] for i in range(len(p))]
    tembus = np.zeros(len(p), bool)
    for langkah in range(1, maks_langkah + 1):
        idx = np.flatnonzero(hidup)
        if len(idx) == 0:
            break
        x = p[idx]
        u1 = interp(U, x)
        s1 = np.linalg.norm(u1, axis=1)
        ok = s1 > 1e-4 * u_rerata
        arahnya = np.zeros_like(u1)
        arahnya[ok] = u1[ok] / s1[ok, None]
        xm = x + 0.5 * h * arahnya
        u2 = interp(U, xm)
        s2 = np.linalg.norm(u2, axis=1)
        ok &= s2 > 1e-4 * u_rerata
        x_baru = x.copy()
        x_baru[ok] = x[ok] + h * u2[ok] / s2[ok, None]
        p[idx] = x_baru
        keluar = (x_baru < 0).any(axis=1) | (x_baru > n - 1).any(axis=1)
        sampai = x_baru[:, a] >= n - 4
        mati = ~ok | keluar | sampai
        tembus[idx[sampai & ok]] = True
        if langkah % simpan_tiap == 0 or mati.any():
            for j, i in enumerate(idx):
                if langkah % simpan_tiap == 0 or mati[j]:
                    jejak[i].append(x_baru[j].copy())
                    laju[i].append(float(s2[j]) if ok[j] else 0.0)
        hidup[idx[mati]] = False
    terpilih = [i for i in range(len(p)) if tembus[i] and len(jejak[i]) >= 4][:n_garis]
    return [np.array(jejak[i]) for i in terpilih], [np.array(laju[i]) for i in terpilih], \
        u_rerata, int(tembus.sum()), len(p)


def tulis_garis(berkas, garis, laju, u_rerata):
    awal = np.zeros(len(garis) + 1, np.uint32)
    awal[1:] = np.cumsum([len(g) for g in garis])
    pos = np.clip(np.round(np.concatenate(garis) * SKALA_U16), 0, 65535).astype(np.uint16)
    s = np.concatenate(laju)
    lg = np.log10(np.maximum(s, 1e-12) / u_rerata)
    kode = np.clip(np.round((lg - LOG_MIN) / (LOG_MAX - LOG_MIN) * 255), 0, 255).astype(np.uint8)
    simpan(berkas, awal.tobytes() + pos.tobytes() + kode.tobytes())
    return int(awal[-1])


def tulis_mesh(berkas, v, fc):
    pos = np.clip(np.round(v * SKALA_U16), 0, 65535).astype(np.uint16)
    tipe = np.uint16 if len(v) < 65536 else np.uint32
    isi = np.array([len(v), len(fc), 2 if tipe is np.uint16 else 4], np.uint32).tobytes() + pos.tobytes()
    isi += b"\0" * ((4 - len(isi) % 4) % 4)
    simpan(berkas, isi + fc.astype(tipe).tobytes())


def simpan(berkas, isi):
    """Larik biner little-endian disimpan sebagai skrip .js berisi string base64. Skrip dapat dimuat dengan
    tag <script> sehingga penampil juga berjalan bila index.html dibuka langsung (file://) tanpa server,
    dan hosting artifact menyajikan .js (bukan .bin)."""
    kunci = os.path.splitext(os.path.basename(berkas))[0]
    with open(berkas, "w", encoding="ascii") as f:
        f.write(f'(window.DATA_3D = window.DATA_3D || {{}})["{kunci}"] = "')
        f.write(base64.b64encode(isi).decode("ascii"))
        f.write('";\n')


def info_batuan(b):
    pk = baca_json(HASIL, b, "permeabilitas_kondisi_tunak.json")
    leher = baca_json(HASIL, b, "karakteristik_leher_pori.json")
    nama = {e["batuan"]: e["nama"] for e in baca_json(KERJA, "data", "sumber", "metadata-citra.json")["berkas"]}
    info = {"batuan": b, "nama": nama.get(b, pk["nama"]), "phi_total": pk["phi_total"], "phi_tersambung": pk["phi_tersambung"],
            "k_mD": pk["K_mD"], "k_geo_mD": pk["K_geometrik_mD"], "anisotropi": pk["anisotropi_k_max_min"],
            "dt_um": leher["dt_um"], "dt_per_dx": leher["dt_um"] / 2.25, "laporan": b in LAPORAN}
    if b in LAPORAN:
        kr = {r["cara"]: r for r in baca_json(HASIL, b, "rekap_tingkat_kritis_fov.json")}
        m = kr["majority"]
        info["kritis"] = {"dx_um": m["dx_kritis_nyquist_um"], "fov_volume": m["lipatan_fov_nyquist_volume"],
                          "pita_persen": m["setengah_lebar_segmentasi_persen"]}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batuan", nargs="+", default=BATUAN)
    ap.add_argument("--arah", nargs="+", default=["x", "y", "z"])
    ap.add_argument("--garis", type=int, default=400)
    ap.add_argument("--benih", type=int, default=2026)
    arg = ap.parse_args()
    os.makedirs(KELUAR, exist_ok=True)
    meta_berkas = os.path.join(KELUAR, "meta.json")
    meta = baca_json(meta_berkas) if os.path.exists(meta_berkas) else {"batuan": {}}
    meta.update({"N": N, "dx_um": 2.25, "skala_u16": SKALA_U16, "log_laju": [LOG_MIN, LOG_MAX],
                 "urutan": BATUAN})
    for b in arg.batuan:
        t0 = time.time()
        img = baca_citra(b)
        info = info_batuan(b)
        assert abs((img == 0).mean() - info["phi_total"]) < 1e-6, f"{b}: porositas citra tidak cocok dengan hasil"
        v, fc = mesh_pori(img)
        tulis_mesh(os.path.join(KELUAR, f"{b}_pori.js"), v, fc)
        info["mesh"] = {"simpul": len(v), "segitiga": len(fc)}
        info["arah"] = {}
        for arah in arg.arah:
            # periksa pendamping: k dari medan tersimpan sama dengan hasil maraton
            pend = baca_json(MEDAN, f"{b}_T0_{arah}.json")
            assert pend["cocok"] and abs(pend["K_mD_tersimpan"] / info["k_mD"][arah] - 1) < 1e-9, (b, arah)
            U, n_fluida = medan_padat(b, arah)
            rng = np.random.default_rng(arg.benih)
            garis, laju, u_rerata, n_tembus, n_benih = streamline(U, arah, arg.garis, rng)
            n_titik = tulis_garis(os.path.join(KELUAR, f"{b}_{arah}.js"), garis, laju, u_rerata)
            info["arah"][arah] = {"garis": len(garis), "titik": n_titik, "u_rerata_kisi": u_rerata,
                                  "benih": n_benih, "tembus": n_tembus, "langkah_lbm": pend["langkah"]}
            del U
            print(f"  {b} {arah}: {len(garis)} garis, {n_titik} titik ({n_tembus}/{n_benih} benih tembus)", flush=True)
        meta["batuan"][b] = info
        with open(meta_berkas, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        with open(os.path.join(KELUAR, "meta.js"), "w", encoding="utf-8") as f:
            f.write("window.META_3D = " + json.dumps(meta, ensure_ascii=False) + ";\n")
        print(f"{b}: mesh {len(v)} simpul, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
