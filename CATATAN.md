# Catatan rinci repositori

Pelengkap [README.md](README.md): nilai setiap angka laporan beserta program dan berkasnya, susunan folder,
keadaan yang perlu diketahui pembaca, catatan menjalankan, dan mesin yang dipakai.

## Catatan menjalankan

- Pada mesin tanpa CUDA (misalnya macOS), `cupy-cuda13x` tidak dapat dipasang; pasang tanpa dua baris CUDA:
  `grep -v -E '^(cupy-cuda13x|cuda-pathfinder)' lingkungan.txt > lingkungan-cpu.txt && .venv/bin/pip install -r lingkungan-cpu.txt`.
  Langkah 1 dan 2 pada README tetap berjalan apa adanya (langkah 09 hanya membaca hasil tersimpan; diuji pada macOS, Apple M4 Pro, Python 3.14).
- Tanpa GPU NVIDIA, ganti `--backend gpu` dengan `--backend cpu`. Kedua jalur memakai rumus yang sama dari
  `program/inti/` dan memberi permeabilitas yang sama sampai selisih nisbi 4×10⁻¹².
- `--out_dir hasil_ulang` penting pada langkah 3 README: tanpa itu, berkas kasus yang sudah tunak di `hasil/` dipakai ulang.
- `program/run.py` tanpa argumen membuka menu interaktif (backend, langkah, batuan, folder keluaran).
- Batuan lain dari paket data yang sama dapat dijalankan dengan `--rocks Berea BUG` atau `--rocks all`;
  hanya citra batuan yang dipilih yang diunduh.

## Daftar angka pada laporan beserta nilai dan asalnya

Semua berkas keluaran di bawah ada di repositori. Angka backend GPU dikutip laporan; `hasil/cpu/` memuat
hasil CPU yang sama untuk pemeriksaan paritas. `<B>` = `Bentheimer` atau `BSG`.

| Angka pada laporan | Nilai (Bentheimer / BSG) | Program yang menghasilkan | Berkas keluaran |
|---|---|---|---|
| Porositas citra φ (Tabel 1, A4, A5) | 0,264 / 0,187 | `program/gpu/gpu_01_permeabilitas_resolusi_penuh.py` | `hasil/gpu/<B>/permeabilitas_kondisi_tunak.json` |
| Selisih φ terhadap tabel acuan (Bagian 2.1) | 2×10⁻⁷ / 4×10⁻⁷ | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| k_geo data acuan (Tabel 1) | 3.137 / 574 mD | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Leher penentu d_t dan d_t/Δx (Tabel 1, 5, A3) | 22,5 / 13,5 µm; 10,0 / 6,0 | `program/gpu/gpu_03_sebaran_leher_pori.py` | `hasil/gpu/<B>/tabel_karakteristik_leher_pori.csv`, `tabel_evaluasi_kelayakan_resolusi.csv` |
| Waktu rantai kerja GPU dan CPU (Bagian 2.3, Lampiran A) | 40 dan 278 / 116 dan 794 menit | catatan waktu produksi `program/run.py` | `hasil/waktu_maraton.csv` |
| MLUPS dan memori (Lampiran A) | ± 850 / 130 MLUPS; 1,4–1,9 GB | `program/inti/inti_08_komparasi_cpu_gpu.py` | `hasil/tabel_komparasi_cpu_gpu.csv` |
| Selisih k GPU–CPU terbesar (Bagian 2.3) | 4×10⁻¹² (120 kasus) | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Beban proses lain terbesar (Lampiran A) | 0,4 / 1,5 % | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Galat saluran persegi, orde, GCI (Tabel 2) | 0,14 % pada 32 voxel; 1,93; 0,20 % | `program/gpu/gpu_00_verifikasi_pemecah.py` | `hasil/gpu/verifikasi/tabel_verifikasi.csv`, `verifikasi_pemecah.json` |
| k per arah, langkah, simpangan fluks (Tabel 3) | k_x 2.965 / 696 mD, … | `program/gpu/gpu_01_permeabilitas_resolusi_penuh.py` | `hasil/gpu/<B>/tabel_permeabilitas_tunak.csv`, `simulasi_multiresolusi/<B>_T0_resolusi_penuh_<arah>.json` |
| τ_u, Mach, Reynolds pori (Tabel 3, Bagian 5) | τ_u 1,42 / 1,50 (arah x), … | `program/gpu/gpu_05_multiresolusi_lbm.py` | `hasil/gpu/<B>/tabel_sifat_efektif.csv` |
| Selisih k terhadap acuan, langkah fluks, kelipatan langkah (Bagian 3, 5, Lampiran B) | −0,13 sampai −0,42 %; 1,6–2,8 kali | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Ambang T dan φ pada tujuh geseran (Tabel 4) | — | `program/inti/inti_02_rentang_segmentasi.py` | `hasil/gpu/<B>/tabel_rentang_segmentasi.csv` |
| Δ_seg, rentang k_geo, median, kepekaan (Bagian 6) | ±33,0 / ±52,2 %; 2.291–4.369 / 347–957 mD | `program/inti/inti_02_rentang_segmentasi.py` | `hasil/gpu/<B>/basis_acuan.json` |
| Asimetri rentang, langkah terbanyak sapuan (Bagian 6) | +39 / −27 %; 166.500 langkah | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Kelayakan per tingkat (Tabel 5) | — | `program/inti/inti_06_evaluasi_kelayakan_resolusi.py` | `hasil/gpu/<B>/tabel_evaluasi_kelayakan_resolusi.csv` |
| L_REV dan koefisien variasi k (Bagian 8) | 160 / 192 voxel; 9 / 57 % | `program/inti/inti_04_volume_elementer.py` | `hasil/gpu/<B>/volume_elementer.json`, `tabel_volume_elementer.csv` |
| Sifat empat tingkat, dua cara (Tabel A4, A5) | — | `program/gpu/gpu_05_multiresolusi_lbm.py` | `hasil/gpu/<B>/tabel_dua_cara_penurunan.csv` |
| Simpangan per arah dan k_geo per tingkat (Bagian 10) | — | `program/inti/inti_06_evaluasi_kelayakan_resolusi.py` | `hasil/gpu/<B>/tabel_simpangan_permeabilitas.csv` |
| Tingkat kritis, simpangan, medan pandang (Tabel 6) | 9 µm, 64× / 4,5 µm, 8× | `program/inti/inti_06_evaluasi_kelayakan_resolusi.py` | `hasil/gpu/<B>/tabel_tingkat_kritis_fov.csv` |
| Tingkat runtuh tiga sifat dan pitanya (Tabel 7) | — | `program/inti/inti_07_rekapitulasi_visualisasi.py` | `hasil/gpu/<B>/rekapitulasi_akhir_s4.json`, `tabel_komparasi_tiga_sifat.csv` |
| Anggaran ketidakpastian u, U, bias (Tabel 8) | U 29,7 / 108 % | `program/inti/inti_07_rekapitulasi_visualisasi.py` | `hasil/gpu/<B>/tabel_anggaran_ketidakpastian.csv`, `rekapitulasi_akhir_s4.json` |
| Selang 95 % k resolusi penuh (Bagian 12) | 2.300–4.200 / 190–1.700 mD | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| k laboratorium, φ formasi, r̄_t (Tabel A2, A3) | 386 / 80 mD | masukan dari Dharmawan dkk. (2026) | `data/pembanding_eksternal.csv` |
| k Koponen dan nisbah antarbatuan (Tabel A2) | 3.945 / 847 mD; 5,5 / 4,8 / 4,7 | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Nisbah 2 r̄_t/Δx (Bagian 10) | 1,98 pada 9 µm / 2,8 pada 4,5 µm | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Gambar 1–5 dan A1 | — | `program/buat_gambar.py` (langkah 09) | `gambar/gambar_*.png` |

Waktu rantai kerja dicatat dari keluaran `program/run.py` saat produksi ("Selesai Langkah … dalam … detik"),
karena waktu berbeda pada setiap mesin; durasi tiap kasus juga tersimpan di `durasi_detik` pada berkas kasus.

## Susunan folder

| Folder | Isi |
|---|---|
| `data/` | `metadata.json` (sumbu, arah aliran, prapemrosesan, ambang, sidik SHA-256 dan ID Google Drive citra), data acuan dari bundel penugasan di `data/sumber/`, dan `pembanding_eksternal.csv`. Citra `.raw` diunduh ke `data/sumber/citra-biner/` dari [folder Google Drive publik](https://drive.google.com/drive/folders/1MGmvIBDEevMjF6MFDZj1D5Wtg2bm5vne) (subfolder `citra-biner`), atau secara manual dengan `.venv/bin/python program/unduh_data.py [--rocks all]`. |
| `program/` | `run.py` (peluncur), langkah 00–08 per backend di `gpu/` dan `cpu/`, rumus bersama di `inti/`, `angka_laporan.py` dan `buat_gambar.py` (langkah 09), `unduh_data.py`, uji rumus (`inti/inti_09_uji_rumus.py`), dan audit (`inti/inti_10_audit_hasil.py`) |
| `hasil/` | Keluaran berangka per backend dan per batuan (`hasil/gpu/<B>/`, `hasil/cpu/<B>/`): tabel CSV, ringkasan JSON, dan setiap berkas kasus LBM beserta riwayat konvergensinya |
| `gambar/` | Gambar laporan, dibuat oleh `program/buat_gambar.py` |

Langkah program: 00 verifikasi pemecah, 01 permeabilitas resolusi penuh, 02 rentang segmentasi, 03 leher pori,
04 volume elementer, 05 penurunan resolusi dan LBM, 06 tingkat kritis, 07 rekapitulasi tiga sifat dan
ketidakpastian, 08 komparasi GPU–CPU, 09 angka dan gambar laporan.

## Keadaan yang perlu diketahui pembaca

- **Simulasi yang belum konvergen:** tidak ada. Seluruh 120 kasus kedua batuan (per batuan: 3 resolusi penuh,
  18 kasus T1–T3, 21 sapuan segmentasi, 18 subvolume) tunak menurut sisa |ΔK/K| < 10⁻⁶ dan simpangan fluks
  antaririsan < 10⁻³ pada dua pemeriksaan berturut-turut. Data acuan sendiri memakai kriteria yang lebih longgar
  (sisa < 5×10⁻⁵): acuan BSG arah x dan y berhenti saat simpangan fluksnya masih 1,2 dan 1,9×10⁻³.
- **Volume elementer yang tidak tercapai:** REV permeabilitas tidak tercapai sampai subvolume 128³ pada kedua
  batuan (koefisien variasi 9 % dan 57 %). REV porositas dan luas spesifik tercapai pada 160 voxel (Bentheimer)
  dan 192 voxel (BSG). Ketidakpastian keterwakilan ini masuk ke anggaran ketidakpastian (Tabel 8), tetapi
  tidak ke simpangan antartingkat karena semua tingkat memakai domain fisis yang sama.
- **Percontoh yang dikeluarkan beserta alasannya:** Bentheimer dan BSG dipilih sebelum simulasi, karena leher
  porinya kontras (22,5 dan 13,5 µm) dan data acuannya lengkap di ketiga arah. Sembilan batuan lain dari paket
  data yang sama juga dijalankan dengan program ini untuk bahan visualisasi tiga dimensi, tetapi tidak dibahas
  di laporan dan hasilnya tidak disertakan di repositori. Hasilnya dapat dibuat ulang dengan `--rocks all`.
  Parker termasuk yang dijalankan; data acuannya belum konvergen dan 20 dari 60 kasusnya tidak tembus pada
  tingkat kasar.
- **Hasil yang tidak sesuai harapan:**
  - Permeabilitas simulasi 7 sampai 8 kali permeabilitas laboratorium sumbat induk (3.127 lawan 386 mD dan
    573 lawan 80 mD), padahal porositasnya dekat dengan rerata formasi. Penyebabnya belum dapat dipastikan;
    dugaan terkuat adalah perbedaan skala antara jendela 576 µm dan sumbat. Nisbah antarbatuan tetap sejalan
    (5,5 lawan 4,8). Tingkat kritis dihitung dari simpangan nisbi terhadap resolusi penuh, bukan dari nilai mutlak k.
  - Mulai T2 pengkasaran menaikkan k pada kedua cara penurunan resolusi, padahal pada T1 suara terbanyak
    menurunkannya sekitar 6 %.
  - BSG dengan suara terbanyak lolos pita sampai 18 µm karena simpangannya tidak monoton (+32,9 % di 9 µm,
    +29,8 % di 18 µm), walaupun lehernya tinggal 0,75 voxel. Karena itu tingkat kritis juga disyaratkan memenuhi
    batas Nyquist (d_t/Δx ≥ 2).
  - Bila leher diukur dengan diameter leher rerata jaringan pori, Bentheimer pada 9 µm hanya 1,98 voxel, tepat di
    tepi batas Nyquist; 9 µm untuk Bentheimer sebaiknya dibaca sebagai batas atas.
  - Permeabilitas resolusi penuh berselisih −0,13 sampai −0,42 % dari data acuan. Selisih ini bukan akibat
    kriteria berhenti, karena pada langkah saat acuan berhenti k sudah berselisih −0,10 sampai −0,39 %.

## Lingkungan dan mesin

Versi pustaka yang dipakai tercatat pada `lingkungan.txt`, dihasilkan dengan `pip freeze` pada lingkungan
bersih (Python 3.13.5). Backend GPU memakai `cupy-cuda13x` dan memerlukan toolkit CUDA 13; backend CPU
memakai Numba dan mengunci thread ke inti fisik.

| Komponen | Mesin laporan |
|---|---|
| GPU | NVIDIA A30, 24 GB, CUDA 13 |
| CPU | 2 × Intel Xeon Gold 5320 2,2 GHz, 52 inti fisik |
| Memori | 377 GB |
| Sistem operasi | Debian GNU/Linux 13, kernel 6.12 |

Sumber citra: Neumann, R. F., dkk. (2021). Multiscale microstructures and fluid flow properties of sandstones.
*Scientific Reports*. doi:10.1038/s41598-021-90090-0
