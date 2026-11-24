# Batas Kekasaran Citra: Menakar Seberapa Kasar Citra Mikrotomografi Masih Berguna untuk Fisika Batuan Komputasional

**Nama:** Dani Hadimukti · **NPM:** 251020260014 · **Jalur:** Simulasi · **Proyek:** S4 Batas Kekasaran Citra  
**Percontoh:** Bentheimer dan Berea Sister Gray (BSG), citra biner 256³ voxel, Δx = 2,25 µm (Neumann dkk., 2021)

## Pertanyaan yang dijawab

Bila citra diturunkan resolusinya dari 2,25 µm menjadi 4,5; 9; dan 18 µm, pada ukuran voxel berapa simpangan
permeabilitas mulai melampaui setengah lebar rentang yang ditimbulkan segmentasi pada resolusi penuh?
**Jawaban:** Bentheimer sampai 9 µm (medan pandang 64 kali volume) dan BSG sampai 4,5 µm (8 kali volume):
simpangan k_geo di dalam pita segmentasi (±33,0 % dan ±52,2 %) dan leher penentu terwakili sedikitnya 2 voxel.

## Cara menjalankan

Diuji dari mesin bersih (Debian 13, Python 3.13.5; juga macOS, Python 3.14). Citra diunduh sendiri oleh `program/run.py`.

```sh
git clone https://github.com/dansek/resolusi-citra-batuan.git && cd resolusi-citra-batuan
python3 -m venv .venv && .venv/bin/pip install -r lingkungan.txt   # tanpa CUDA: lihat CATATAN.md
.venv/bin/python program/inti/inti_09_uji_rumus.py                  # 1) uji rumus, harus "SEMUA UJI LULUS"
.venv/bin/python program/run.py --non-interactive --backend gpu --steps 9          # 2) angka dan gambar laporan
.venv/bin/python program/inti/inti_10_audit_hasil.py --backend gpu --banding cpu   #    harus "AUDIT BERSIH"
.venv/bin/python program/run.py --non-interactive --backend gpu --steps 0 1 2 3 4 5 6 7 9 \
    --rocks Bentheimer BSG --mode_t0 simulasi --out_dir hasil_ulang                # 3) hitung ulang semuanya
.venv/bin/python program/inti/inti_10_audit_hasil.py --out_dir hasil_ulang --backend gpu
```

Tanpa GPU NVIDIA, ganti `--backend gpu` dengan `--backend cpu` (hasil sama sampai selisih nisbi 4×10⁻¹²).
**Perkiraan waktu hitung seluruhnya** (langkah 01–07, dua batuan): 2,6 jam pada GPU NVIDIA A30 atau 17,9 jam pada
CPU 2 × Intel Xeon Gold 5320; langkah 1 dan 2 di atas masing-masing sekitar satu menit.

## Daftar angka pada laporan beserta asalnya

`<B>` = `Bentheimer` atau `BSG`. Nilai setiap angka dan rincian per baris ada di [CATATAN.md](CATATAN.md).

| Angka pada laporan | Program | Berkas keluaran |
|---|---|---|
| Verifikasi pemecah (Tabel 2, Gambar 1) | `program/gpu/gpu_00_verifikasi_pemecah.py` | `hasil/gpu/verifikasi/` |
| φ, k per arah, langkah, simpangan fluks (Tabel 1, 3) | `program/gpu/gpu_01_permeabilitas_resolusi_penuh.py` | `hasil/gpu/<B>/tabel_permeabilitas_tunak.csv` |
| Rentang segmentasi Δ_seg (Tabel 4, Bagian 6, Gambar 2) | `program/inti/inti_02_rentang_segmentasi.py` | `hasil/gpu/<B>/tabel_rentang_segmentasi.csv`, `basis_acuan.json` |
| Leher penentu d_t (Tabel 1, 5, A3) | `program/gpu/gpu_03_sebaran_leher_pori.py` | `hasil/gpu/<B>/tabel_karakteristik_leher_pori.csv` |
| Volume elementer (Bagian 8, Gambar 3) | `program/inti/inti_04_volume_elementer.py` | `hasil/gpu/<B>/tabel_volume_elementer.csv` |
| Empat tingkat, dua cara, τ_u, Mach, Re (Tabel A4, A5, Bagian 5) | `program/gpu/gpu_05_multiresolusi_lbm.py` | `hasil/gpu/<B>/tabel_dua_cara_penurunan.csv`, `tabel_sifat_efektif.csv` |
| Kelayakan dan tingkat kritis (Tabel 5, 6, Gambar 4) | `program/inti/inti_06_evaluasi_kelayakan_resolusi.py` | `hasil/gpu/<B>/tabel_tingkat_kritis_fov.csv` |
| Tiga sifat dan anggaran ketidakpastian (Tabel 7, 8, Gambar 5) | `program/inti/inti_07_rekapitulasi_visualisasi.py` | `hasil/gpu/<B>/rekapitulasi_akhir_s4.json` |
| MLUPS dan memori (Lampiran A) | `program/inti/inti_08_komparasi_cpu_gpu.py` | `hasil/tabel_komparasi_cpu_gpu.csv` |
| Angka turunan lain: GPU–CPU, selisih acuan, langkah, selang 95 %, Koponen, nisbah | `program/angka_laporan.py` | `hasil/angka_laporan.csv` |
| Waktu rantai kerja (Bagian 2.3, Lampiran A) | catatan waktu produksi `program/run.py` | `hasil/waktu_maraton.csv` |
| k laboratorium, φ formasi, leher jaringan pori (Tabel A2, A3) | masukan dari Dharmawan dkk. (2026) | `data/pembanding_eksternal.csv` |
| Gambar 1–5 dan A1 | `program/buat_gambar.py` (langkah 09) | `gambar/` |

## Susunan folder dan keadaan yang perlu diketahui

`data/` metadata citra dan data acuan · `program/` satu berkas per langkah (00–09) · `hasil/` CSV dan JSON per
backend dan batuan · `gambar/` gambar laporan · `lingkungan.txt` versi pustaka (`pip freeze`).

- **Belum konvergen:** tidak ada; 120 dari 120 kasus tunak.
- **REV tidak tercapai:** permeabilitas sampai subvolume 128³ (koefisien variasi 9 % dan 57 %).
- **Percontoh dikeluarkan:** sembilan batuan lain hanya untuk visualisasi tiga dimensi, tidak di laporan.
- **Tidak sesuai harapan:** k simulasi 7 sampai 8 kali k laboratorium; BSG dengan suara terbanyak lolos pita
  sampai 18 µm walaupun lehernya 0,75 voxel. Rinciannya di [CATATAN.md](CATATAN.md).

**Alat bantu kecerdasan buatan:** Claude Code (Anthropic, Claude Opus 5.5) dipakai untuk program, grafik, draf,
dan bahasa laporan serta slide. Pilihan batuan, kriteria, dan penerimaan hasil ditetapkan penulis; seluruh
angka dihitung oleh program di repositori ini.
