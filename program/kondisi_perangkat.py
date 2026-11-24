"""
program/kondisi_perangkat.py
============================
Bukti bahwa durasi sebuah run LBM diukur pada perangkat yang tidak dipakai proses lain (perbandingan waktu adil).

mulai(perangkat): dipanggil tepat sebelum pengukuran waktu dimulai. Mengukur beban CPU sistem selama 1 detik
                  (/proc/stat di Linux, psutil di sistem lain; run ini belum berjalan) dan, untuk GPU, proses komputasi lain di kartu.
selesai(snap)   : dipanggil tepat setelah pengukuran waktu berhenti. Beban proses LAIN selama run
                  = (jiffy sibuk seluruh sistem - waktu CPU proses ini) / jiffy total; GPU diperiksa lagi.
menganggur = beban awal < 5 % DAN beban proses lain selama run < 5 % kapasitas CPU DAN (GPU) tak ada proses lain.
Catatan: beban rata-rata 1 menit (loadavg) tidak dipakai karena masih memuat sisa run sebelumnya dari proses ini.
"""

import os
import time
import subprocess

AMBANG = 0.05


def _stat():
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]      # total, idle + iowait (jiffy)
    except FileNotFoundError:           # macOS/Windows: tanpa /proc, pakai psutil (detik -> jiffy)
        import psutil
        c = psutil.cpu_times()
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        return sum(c) * hz, (c.idle + getattr(c, "iowait", 0.0)) * hz


def _cpu_proses():
    t = os.times()
    return (t.user + t.system) * os.sysconf("SC_CLK_TCK")


def _gpu_lain():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return len([q for q in out if q.strip() and int(q) != os.getpid()])
    except Exception:
        return None


def mulai(perangkat):
    t0, i0 = _stat()
    time.sleep(1.0)
    t1, i1 = _stat()
    snap = {"perangkat": perangkat, "beban_cpu_awal": 1.0 - (i1 - i0) / max(t1 - t0, 1),
            "_total": t1, "_idle": i1, "_proses": _cpu_proses()}
    if perangkat == "gpu":
        snap["gpu_proses_lain_awal"] = _gpu_lain()
    return snap


def selesai(snap):
    t2, i2 = _stat()
    total = max(t2 - snap["_total"], 1)
    sibuk = total - (i2 - snap["_idle"])
    sendiri = _cpu_proses() - snap["_proses"]
    hasil = {k: v for k, v in snap.items() if not k.startswith("_")}
    hasil["beban_cpu_lain_selama_run"] = max(0.0, (sibuk - sendiri) / total)
    hasil["jumlah_cpu_logis"] = os.cpu_count()
    ok = hasil["beban_cpu_awal"] < AMBANG and hasil["beban_cpu_lain_selama_run"] < AMBANG
    if snap["perangkat"] == "gpu":
        hasil["gpu_proses_lain_akhir"] = _gpu_lain()
        g = (hasil.get("gpu_proses_lain_awal"), hasil["gpu_proses_lain_akhir"])
        ok = ok and g == (0, 0)
    hasil["menganggur"] = bool(ok)
    hasil["ambang"] = AMBANG
    return hasil
