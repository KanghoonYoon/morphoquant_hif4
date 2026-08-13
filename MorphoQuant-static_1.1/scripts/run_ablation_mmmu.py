#!/usr/bin/env python3
"""
Ablation study: Grid-test calib_size × search_size for InternVL2.5-8B Morpho+HiF4 (W4A4) on MMMU.

Usage:
    python scripts/run_ablation_mmmu.py \
        --base-config configs/internvl2.5-8b/morpho_withhif/mmmu_morpho_withhif4.yaml \
        --calib-sizes 16,32,64,128,256 \
        --search-sizes 4,8,16,32,64 \
        --gpus 4,5,6 \
        --base-save-dir /private/wy/logs/MorphoQuant/ablation_calib_search
"""

import argparse
import csv
import os
import queue
import subprocess
import sys
import time
import yaml
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
TEMP_CONFIG_DIR = os.path.join(SCRIPTS_DIR, ".ablation_configs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study: calib_size × search_size grid for MMMU"
    )
    parser.add_argument(
        "--base-config",
        default="configs/internvl2.5-8b/morpho_withhif/mmmu_morpho_withhif4.yaml",
        help="Base YAML config to clone from (default: internvl2.5-8b morpho_withhif4 MMMU)",
    )
    parser.add_argument(
        "--calib-sizes",
        default="16,32,64,128,256",
        help="Comma-separated calib_size values (default: 16,32,64,128,256)",
    )
    parser.add_argument(
        "--search-sizes",
        default="4,8,16,32,64",
        help="Comma-separated search_size values (default: 4,8,16,32,64)",
    )
    parser.add_argument(
        "--gpus",
        default="4,5,6",
        help="Comma-separated GPU IDs for parallel execution (default: 4,5,6)",
    )
    parser.add_argument(
        "--base-save-dir",
        default="/private/wy/logs/MorphoQuant/ablation_calib_search",
        help="Base directory for per-experiment results (default: /private/wy/logs/MorphoQuant/ablation_calib_search)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned commands, do not execute",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess output in real-time instead of logging to files",
    )
    return parser.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def generate_configs(
    base_config_path: str,
    calib_sizes: List[int],
    search_sizes: List[int],
    base_save_dir: str,
) -> List[Dict]:
    """Generate per-experiment config files and return a list of job descriptors."""
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

    # Read base config as raw YAML text to preserve comments and formatting
    with open(os.path.join(PROJECT_DIR, base_config_path), "r") as f:
        base_yaml = yaml.safe_load(f)

    jobs = []
    for calib_size in calib_sizes:
        for search_size in search_sizes:
            # Clone and modify config
            cfg = yaml.safe_load(yaml.dump(base_yaml))  # deep copy via serialization
            cfg["quant"]["calib_size"] = calib_size
            cfg["quant"]["search_size"] = search_size

            # Unique save directory per experiment
            exp_name = f"calib{calib_size}_search{search_size}"
            save_dir = os.path.join(base_save_dir, exp_name)
            cfg["run"]["save_dir"] = save_dir

            # Write temp config
            config_filename = f"mmmu_ablation_{exp_name}.yaml"
            config_path = os.path.join(TEMP_CONFIG_DIR, config_filename)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

            jobs.append({
                "calib_size": calib_size,
                "search_size": search_size,
                "exp_name": exp_name,
                "config_path": config_path,
                "save_dir": save_dir,
                "log_path": os.path.join(TEMP_CONFIG_DIR, f"{exp_name}.log"),
            })

    return jobs


def run_one_job(job: Dict, gpu_id: int, verbose: bool = False) -> Dict:
    """Run a single experiment on the given GPU. Returns result dict."""
    exp_name = job["exp_name"]
    config_path = job["config_path"]
    log_path = job["log_path"]
    save_dir = job["save_dir"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, "wy_inference_mmmu.py"),
        "--config", config_path,
    ]

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[GPU {gpu_id}] Starting: {exp_name}")
        print(f"[GPU {gpu_id}] Cmd: {' '.join(cmd)}")
        print(f"{'='*60}")
        result = subprocess.run(
            cmd, env=env, cwd=PROJECT_DIR,
            stdout=sys.stdout, stderr=subprocess.STDOUT,
        )
        returncode = result.returncode
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as log_f:
            log_f.write(f"=== {exp_name} | GPU {gpu_id} | {datetime.now()} ===\n")
            log_f.write(f"Cmd: {' '.join(cmd)}\n\n")
            log_f.flush()
            result = subprocess.run(
                cmd, env=env, cwd=PROJECT_DIR,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            returncode = result.returncode

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    status = "OK" if returncode == 0 else f"FAILED (rc={returncode})"

    return {
        **job,
        "gpu_id": gpu_id,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "elapsed_str": f"{hours}h{minutes}m{seconds}s",
    }


def collect_results(jobs: List[Dict]) -> Tuple[Dict, Dict, List[str]]:
    """Scan save_dirs for summary CSVs and build accuracy matrix.

    Returns (matrix, best, errors).
    matrix: {(calib_size, search_size): accuracy}
    best: {"calib_size": ..., "search_size": ..., "accuracy": ...}
    errors: list of error strings
    """
    matrix = {}
    errors = []

    for job in jobs:
        calib_size = job["calib_size"]
        search_size = job["search_size"]
        save_dir = job["save_dir"]

        if job.get("status", "").startswith("FAILED"):
            errors.append(f"{job['exp_name']}: job failed ({job.get('status','')})")
            continue

        if not os.path.isdir(save_dir):
            errors.append(f"{job['exp_name']}: save_dir not found: {save_dir}")
            continue

        # Find summary_report_*.csv
        summary_files = list(Path(save_dir).glob("summary_report_*.csv"))
        if not summary_files:
            errors.append(f"{job['exp_name']}: no summary_report CSV in {save_dir}")
            continue

        # Use the most recent summary file
        summary_path = str(sorted(summary_files)[-1])
        try:
            with open(summary_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("subject", "").strip().upper() == "AVERAGE":
                        acc = float(row["acc"])
                        matrix[(calib_size, search_size)] = acc
                        break
                else:
                    errors.append(f"{job['exp_name']}: no AVERAGE row in {summary_path}")
        except Exception as e:
            errors.append(f"{job['exp_name']}: failed to parse {summary_path}: {e}")

    # Find best
    best = None
    if matrix:
        best_key = max(matrix, key=matrix.get)
        best = {
            "calib_size": best_key[0],
            "search_size": best_key[1],
            "accuracy": matrix[best_key],
        }

    return matrix, best, errors


def print_report(
    calib_sizes: List[int],
    search_sizes: List[int],
    matrix: Dict,
    best: Optional[Dict],
    errors: List[str],
    jobs: List[Dict],
    started_at: datetime,
):
    """Print the final ablation report."""
    print("\n" + "=" * 80)
    print("  ABLATION STUDY REPORT")
    print("=" * 80)
    print(f"  Started:   {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total experiments: {len(jobs)}")
    success_count = sum(1 for j in jobs if j.get("status") == "OK")
    fail_count = len(jobs) - success_count
    print(f"  Successful: {success_count}  |  Failed: {fail_count}")
    print()

    # ---- Per-job status table ----
    print(f"{'Experiment':<28} {'GPU':>4} {'Status':>12} {'Time':>10}")
    print("-" * 58)
    for job in jobs:
        elapsed = job.get("elapsed_str", "-")
        status = job.get("status", "PENDING")
        print(
            f"  {job['exp_name']:<26}  {job.get('gpu_id','-'):>4}  {status:>12}  {elapsed:>10}"
        )
    print()

    # ---- Accuracy matrix ----
    if matrix:
        print("  Accuracy Matrix (calib_size \\ search_size):")
        print()
        # Header
        header = f"{'calib ↓':>10}"
        for ss in search_sizes:
            header += f"  s={ss:>3d}"
        print(header)
        print("-" * len(header))

        for cs in calib_sizes:
            row = f"{'c=' + str(cs):>10}"
            for ss in search_sizes:
                acc = matrix.get((cs, ss))
                if acc is not None:
                    row += f"  {acc:.4f}"
                else:
                    row += f"  {'N/A':>6}"
            print(row)

        print()

        # ---- Best result ----
        if best:
            print(f"  ★ Best: calib_size={best['calib_size']}, "
                  f"search_size={best['search_size']} → "
                  f"accuracy={best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
            print()

        # ---- Trends ----
        print("  Trend by calib_size (averaged over all search_size):")
        for cs in calib_sizes:
            values = [matrix.get((cs, ss)) for ss in search_sizes
                      if matrix.get((cs, ss)) is not None]
            if values:
                avg = sum(values) / len(values)
                print(f"    calib_size={cs:>4d}: avg_acc={avg:.4f}  (n={len(values)})")
        print()
        print("  Trend by search_size (averaged over all calib_size):")
        for ss in search_sizes:
            values = [matrix.get((cs, ss)) for cs in calib_sizes
                      if matrix.get((cs, ss)) is not None]
            if values:
                avg = sum(values) / len(values)
                print(f"    search_size={ss:>4d}: avg_acc={avg:.4f}  (n={len(values)})")
        print()

    # ---- Errors ----
    if errors:
        print(f"  ⚠ Warnings/Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        print()

    print("=" * 80)


def main():
    args = parse_args()
    calib_sizes = parse_int_list(args.calib_sizes)
    search_sizes = parse_int_list(args.search_sizes)
    gpus = parse_int_list(args.gpus)

    if not calib_sizes or not search_sizes or not gpus:
        print("ERROR: calib-sizes, search-sizes, and gpus must all be non-empty.")
        sys.exit(1)

    # Validate GPU IDs
    for g in gpus:
        if not (0 <= g <= 7):
            print(f"WARNING: GPU {g} — ensure this GPU exists and is available.")

    print("=" * 80)
    print("  Ablation Study: calib_size × search_size for MMMU (Morpho+HiF4 W4A4)")
    print("=" * 80)
    print(f"  Base config:    {args.base_config}")
    print(f"  calib_sizes:    {calib_sizes}")
    print(f"  search_sizes:   {search_sizes}")
    print(f"  GPUs:           {gpus}")
    print(f"  Base save dir:  {args.base_save_dir}")
    print(f"  Total jobs:     {len(calib_sizes) * len(search_sizes)}")
    print(f"  Max concurrent: {len(gpus)}")
    if args.dry_run:
        print("  *** DRY RUN MODE ***")
    print()

    # ---- Generate configs ----
    base_config_abs = os.path.join(PROJECT_DIR, args.base_config)
    if not os.path.exists(base_config_abs):
        print(f"ERROR: Base config not found: {base_config_abs}")
        sys.exit(1)

    jobs = generate_configs(
        args.base_config,
        calib_sizes,
        search_sizes,
        args.base_save_dir,
    )
    print(f"Generated {len(jobs)} config files in {TEMP_CONFIG_DIR}")

    if args.dry_run:
        for i, job in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            print(f"\n  [{i+1}/{len(jobs)}] GPU {gpu}: {job['exp_name']}")
            print(f"    Config:  {job['config_path']}")
            print(f"    Save:    {job['save_dir']}")
            print(f"    Log:     {job['log_path']}")
        print("\nDry-run complete. Remove --dry-run to execute.")
        return

    # ---- Run experiments in parallel with GPU queue ----
    started_at = datetime.now()
    max_workers = len(gpus)

    print(f"\nStarting experiments with {max_workers} parallel workers...")
    print(f"Start time: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # GPU queue: each worker acquires a GPU, runs the job, then releases it
    gpu_queue = queue.Queue()
    for g in gpus:
        gpu_queue.put(g)

    gpu_lock = threading.Lock()
    results_lock = threading.Lock()
    results = []
    completed_count = [0]  # mutable counter for thread-safe updates
    total = len(jobs)

    def run_job_with_gpu(job):
        gpu = gpu_queue.get()
        try:
            result = run_one_job(job, gpu, args.verbose)
        finally:
            gpu_queue.put(gpu)

        with results_lock:
            results.append(result)
            completed_count[0] += 1
            completed = completed_count[0]
            status_icon = "✓" if result["status"] == "OK" else "✗"
            print(
                f"  [{completed:>2}/{total}] {status_icon} "
                f"{result['exp_name']:<26} "
                f"GPU={result['gpu_id']}  "
                f"{result['status']:<16}  "
                f"{result['elapsed_str']}"
            )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job_with_gpu, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                # Should not happen since run_job_with_gpu catches everything
                print(f"  UNEXPECTED worker error: {e}")

    # ---- Collect and report ----
    print("\nCollecting results from save directories...")
    matrix, best, errors = collect_results(results)

    print_report(
        calib_sizes, search_sizes, matrix, best, errors,
        results, started_at,
    )

    # Save master summary CSV
    summary_csv = os.path.join(args.base_save_dir, "ablation_master_summary.csv")
    os.makedirs(args.base_save_dir, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["calib_size", "search_size", "accuracy", "status", "elapsed_sec"])
        for job in results:
            acc = matrix.get((job["calib_size"], job["search_size"]))
            writer.writerow([
                job["calib_size"], job["search_size"],
                f"{acc:.6f}" if acc is not None else "",
                job.get("status", ""),
                f"{job.get('elapsed_sec', 0):.1f}",
            ])
    print(f"Master summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
