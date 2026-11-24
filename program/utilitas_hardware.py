#!/usr/bin/env python3
"""
program/utilitas_hardware.py
============================
Utilitas Deteksi dan Konfigurasi Hardware CPU & GPU.
Membedakan secara akurat antara Core Fisik Murni vs Thread Logis (Hyper-Threading / SMT).
Mengonfigurasi afinitas thread proses ke Core Fisik Murni untuk efisiensi komputasi numerik HPC.
"""

import os
import sys
import platform


def deteksi_topologi_cpu():
    """
    Mendeteksi topologi CPU:
    - Membedakan core fisik murni vs thread logis SMT (Hyper-Threading).
    - Menghasilkan daftar ID CPU yang menempati core fisik primer.
    """
    base_path = "/sys/devices/system/cpu"
    core_map = {}
    sockets = set()

    # Metode 1: Membaca sysfs Linux untuk pasangan (physical_package_id, core_id)
    if os.path.isdir(base_path):
        try:
            entries = [e for e in os.listdir(base_path) if e.startswith("cpu") and e[3:].isdigit()]
            entries.sort(key=lambda x: int(x[3:]))
            for cpu_entry in entries:
                cpu_id = int(cpu_entry[3:])
                top_path = os.path.join(base_path, cpu_entry, "topology")
                pkg_file = os.path.join(top_path, "physical_package_id")
                core_file = os.path.join(top_path, "core_id")
                if os.path.exists(pkg_file) and os.path.exists(core_file):
                    with open(pkg_file, "r", encoding="utf-8") as f_pkg, open(core_file, "r", encoding="utf-8") as f_c:
                        pkg_id = f_pkg.read().strip()
                        c_id = f_c.read().strip()
                        key = (pkg_id, c_id)
                        sockets.add(pkg_id)
                        if key not in core_map:
                            core_map[key] = cpu_id
        except Exception:
            pass

    # Ambil nama model CPU
    cpu_name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    total_threads = os.cpu_count() or 1

    if core_map:
        cpu_fisik_ids = sorted(core_map.values())
        core_fisik = len(cpu_fisik_ids)
        threads_per_core = max(1, total_threads // max(1, core_fisik))
    else:
        # Fallback jika bukan Linux atau sysfs tidak dapat diakses
        try:
            import psutil
            c_phys = psutil.cpu_count(logical=False)
            if c_phys:
                core_fisik = c_phys
            else:
                core_fisik = max(1, total_threads // 2) if total_threads > 1 else 1
        except Exception:
            core_fisik = max(1, total_threads // 2) if total_threads > 1 else 1
        cpu_fisik_ids = list(range(core_fisik))
        threads_per_core = max(1, total_threads // core_fisik)

    n_sockets = len(sockets) if sockets else (2 if core_fisik >= 48 else 1)

    return {
        "model": cpu_name,
        "core_fisik": core_fisik,
        "total_threads": total_threads,
        "threads_per_core": threads_per_core,
        "sockets": n_sockets,
        "cpu_fisik_ids": cpu_fisik_ids,
        "smt_aktif": total_threads > core_fisik
    }


def inisialisasi_lingkungan_hardware():
    """
    Inisialisasi dini variabel lingkungan sebelum modul komputasi (Numba, OpenMP, BLAS) diimpor.
    Ini krusial agar Numba membaca batas thread fisik sejak startup tanpa memicu reloader conflict.
    """
    info = deteksi_topologi_cpu()
    c_str = str(info["core_fisik"])
    for var in ["NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if var not in os.environ:
            os.environ[var] = c_str
    if hasattr(os, "sched_setaffinity") and info["cpu_fisik_ids"]:
        try:
            os.sched_setaffinity(0, set(info["cpu_fisik_ids"]))
        except Exception:
            pass
    return info


# Jalankan inisialisasi dini saat modul ini pertama kali diimpor
_TOPOLOGI_SISTEM = inisialisasi_lingkungan_hardware()


def terapkan_afinitas_core_fisik(info=None, verbose=False):
    """
    Mengunci proses komputasi CPU hanya pada Core Fisik Murni (mengabaikan SMT / Hyper-Threading)
    dan mengonfigurasi variabel lingkungan thread (OpenMP, Numba, MKL, BLAS).
    """
    if info is None:
        info = _TOPOLOGI_SISTEM

    core_fisik = info["core_fisik"]
    cpu_fisik_ids = info["cpu_fisik_ids"]

    # Atur thread Numba secara runtime jika modul numba tersedia
    try:
        import numba as nb
        if nb.get_num_threads() != core_fisik:
            nb.set_num_threads(core_fisik)
    except Exception:
        pass

    # Kunci CPU Affinity ke Core Fisik Murni (Linux kernel scheduler)
    sukses_afinitas = False
    if hasattr(os, "sched_setaffinity") and cpu_fisik_ids:
        try:
            os.sched_setaffinity(0, set(cpu_fisik_ids))
            sukses_afinitas = True
        except Exception:
            pass

    if verbose:
        status_af = "Aktif (Core 0 s/d " + str(cpu_fisik_ids[-1]) + ")" if sukses_afinitas else "Variabel Lingkungan"
        print(f"[Hardware CPU] Menggunakan {core_fisik} Core Fisik Murni (dari {info['total_threads']} thread logis) | Afinitas: {status_af}")

    return {
        "core_fisik": core_fisik,
        "sukses_afinitas": sukses_afinitas
    }


if __name__ == "__main__":
    top = deteksi_topologi_cpu()
    print("=" * 60)
    print("HASIL DETEKSI TOPOLOGI HARDWARE CPU:")
    print("=" * 60)
    print(f"Model CPU          : {top['model']}")
    print(f"Socket Fisik       : {top['sockets']}")
    print(f"Core Fisik Murni   : {top['core_fisik']} Core")
    print(f"Thread Logis Total : {top['total_threads']} Thread (SMT: {'Aktif' if top['smt_aktif'] else 'Nonaktif'})")
    print(f"Rasio SMT          : {top['threads_per_core']} Thread per Core Fisik")
    print(f"Rentang Core Fisik : CPU {top['cpu_fisik_ids'][0]} s/d CPU {top['cpu_fisik_ids'][-1]}")
    print("=" * 60)
    af = terapkan_afinitas_core_fisik(top, verbose=True)
