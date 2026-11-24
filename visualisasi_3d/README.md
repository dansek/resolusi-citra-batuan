# Penampil 3D: ruang pori dan aliran LBM pada empat tingkat resolusi

Pelengkap visual laporan, bukan sumber angka laporan. Sebelas batupasir dari paket data yang sama
(Bentheimer dan BSG dibahas di laporan; sembilan lainnya ditandai "belum ditelaah"), pada T0–T3
(2,25; 4,5; 9; 18 µm) dan dua cara penurunan resolusi.

## Cara membuka

**Penampil daring: [https://dansek.github.io/resolusi-citra-batuan/visualisasi_3d/](https://dansek.github.io/resolusi-citra-batuan/visualisasi_3d/)**

Alamat langsung ke batuan, tingkat, cara, dan arah tertentu, misalnya batas kekasaran di laporan:
[Bentheimer 9 µm](https://dansek.github.io/resolusi-citra-batuan/visualisasi_3d/#Bentheimer-T2-majority-x) dan [BSG 4,5 µm](https://dansek.github.io/resolusi-citra-batuan/visualisasi_3d/#BSG-T1-majority-x).

Tanpa internet ke GitHub Pages: klon atau unduh repositori, lalu klik ganda `visualisasi_3d/index.html` di komputer
sendiri (tidak perlu server; Three.js dan huruf IBM Plex tetap dimuat dari internet).

Yang ditampilkan:

- permukaan pori tingkat terpilih, dari volume yang benar-benar disimulasikan
  (`hasil/gpu/<B>/multiresolusi/<B>_<T>_<cara>.npy` pada folder kerja; porositasnya diperiksa terhadap tabel),
- streamline dan partikel dari medan kecepatan LBM tingkat itu sendiri, berwarna menurut laju |u|/ū,
- angka per tingkat, cara, dan arah dari tabel hasil (φ, φ tembus searah aliran, k per arah, k_geo, simpangan,
  Sv, d_t/Δx, langkah LBM), grafik simpangan k_geo terhadap pita segmentasi, dan kesimpulan tingkat kritis.

## Asal data `data/`

Berkas `data/*.js` berisi larik biner little-endian dalam base64 (dimuat lewat tag `<script>` agar
berjalan dari `file://`). Dibangun oleh program di `pembangun/`:

| Program | Isi |
|---|---|
| `pembangun/bangun_data_panel.py` | mesh pori (rerata blok ke grid ≤ 64³, marching cubes pada φ = 0,5), 400 streamline per arah (RK2, langkah 0,5 voxel, benih sebanding fluks di irisan masuk), dan `meta.js` dari tabel hasil |
| `pembangun/bangun_data.py` | fungsi bersama (mesh, streamline, penyimpanan); juga pembangun penampil T0 sebelas batuan |
| `pembangun/ekspor_medan_tingkat_gpu.py` | medan kecepatan T1–T3 di GPU: perulangan pemecah yang sama dengan rantai kerja, dijalankan sampai langkah tunak tersimpan; disimpan hanya bila langkah, k (selisih < 1e-9), dan geometri sama |
| `pembangun/ekspor_medan_tingkat_cpu.py` | versi CPU yang setara (sekitar 90 detik per kasus T2 di laptop) |

Medan kecepatan sendiri (T0 sekitar 1,7 GB dan T1–T3 sekitar 0,4 GB) dan hasil sembilan batuan lain tidak
disimpan di repositori karena ukurannya. Medan T0 diekspor dengan pemecah yang sama dari kasus resolusi penuh;
dari 198 kasus T1–T3, 186 cocok dengan hasil tersimpan dan 12 tidak tembus (Bandera Brown T3 suara terbanyak
arah y dan z, serta sepuluh kasus Parker), sehingga tidak punya aliran.

Membangun ulang (memerlukan folder kerja berisi `hasil/gpu/` sebelas batuan dan `archive/medan-kecepatan*/`):

```sh
KERJA_S4=<folder kerja> KELUAR_PANEL=visualisasi_3d/data python visualisasi_3d/pembangun/bangun_data_panel.py
```
