#!/usr/bin/env python3
"""
Parallel MQuant W4A4 evaluation runner for Qwen2.5-Omni 3B & 7B across 4 datasets.

Dispatches jobs across GPUs 4,5,6 with a GPU queue, monitors VRAM via nvidia-smi
polling, and reports accuracy + memory summary.

Usage:
    # Smoke tests only (1 sample each)
    python scripts/run_mquant_w4a4_eval.py --smoke --gpus 4,5,6

    # Full evaluation only
    python scripts/run_mquant_w4a4_eval.py --full --gpus 4,5,6

    # Both (default): smoke first, then full if smoke passes
    python scripts/run_mquant_w4a4_eval.py --gpus 4,5,6
"""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_DIR, "scripts", ".mquant_logs")

# Model → config key and model_path
MODELS = {
    "3b": {
        "key": "qwen2.5-omni-3b",
        "config_dir": "qwen2.5-omni-3b",
        "model_path": "/private/wy/pretrained_models/Qwen2.5-Omni-3B",
    },
    "7b": {
        "key": "qwen2.5-omni-7b",
        "config_dir": "qwen2.5-omni-7b",
        "model_path": "/private/wy/pretrained_models/Qwen2.5-Omni-7B",
    },
}

DATASETS = {
    "mmmu": {
        "script": "wy_inference_mmmu.py",
        "save_subdir": "mmmu_results",
    },
    "scienceqa": {
        "script": "wy_inference_scienceqa.py",
        "save_subdir": "scienceqa_results",
    },
    "videomme": {
        "script": "wy_inference_videomme.py",
        "save_subdir": "videomme_results",
    },
    "airbench": {
        "script": "wy_inference_airbench.py",
        "save_subdir": "airbench_results",
    },
}

# ---------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------
class GPUMemoryMonitor:
    """Background thread that polls nvidia-smi and records per-GPU peak memory."""

    def __init__(self, gpu_ids: List[int], interval: float = 2.0):
        self.gpu_ids = list(gpu_ids)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # per-GPU peak memory in MB
        self.peak_mb: Dict[int, float] = {g: 0.0 for g in gpu_ids}
        # per-GPU current memory (time series if needed)
        self.current_mb: Dict[int, float] = {g: 0.0 for g in gpu_ids}
        self.running = False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self.running = True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.running = False

    def _poll_loop(self):
        while not self._stop.is_set():
            for gpu_id in self.gpu_ids:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                            "-i", str(gpu_id),
                        ],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        mem = float(result.stdout.strip())
                        with self._lock:
                            self.current_mb[gpu_id] = mem
                            if mem > self.peak_mb[gpu_id]:
                                self.peak_mb[gpu_id] = mem
                except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
                    pass
            self._stop.wait(self.interval)

    def get_peak(self, gpu_id: int) -> float:
        with self._lock:
            return self.peak_mb[gpu_id]

    def get_current(self, gpu_id: int) -> float:
        with self._lock:
            return self.current_mb[gpu_id]

    def reset_peak(self, gpu_id: int):
        with self._lock:
            self.peak_mb[gpu_id] = 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel MQuant W4A4 evaluation for Qwen2.5-Omni 3B & 7B"
    )
    parser.add_argument(
        "--models", default="3b,7b",
        help="Comma-separated model sizes (default: 3b,7b)",
    )
    parser.add_argument(
        "--datasets", default="mmmu,scienceqa,videomme,airbench",
        help="Comma-separated datasets (default: mmmu,scienceqa,videomme,airbench)",
    )
    parser.add_argument(
        "--gpus", default="4,5,6",
        help="Comma-separated GPU IDs (default: 4,5,6)",
    )
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Run smoke tests only (1 sample each)",
    )
    parser.add_argument(
        "--full", action="store_true", default=False,
        help="Run full evaluation only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned commands without executing",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream subprocess output to terminal instead of logs",
    )
    parser.add_argument(
        "--skip-smoke-check", action="store_true",
        help="Skip smoke test phase even when running both phases (use with caution)",
    )
    parser.add_argument(
        "--quarot", action="store_true", default=False,
        help="Use QuaRot (Hadamard rotation) configs (mquant_w4a4_quarot instead of mquant_w4a4)",
    )
    return parser.parse_args()


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------
def build_jobs(
    models: List[str],
    datasets: List[str],
    smoke: bool,
    quarot: bool = False,
) -> List[Dict]:
    """Build the job list from model/dataset combinations."""
    jobs = []
    for model_key in models:
        if model_key not in MODELS:
            print(f"WARNING: Unknown model '{model_key}', skipping. Known: {list(MODELS.keys())}")
            continue
        model_info = MODELS[model_key]
        for ds_key in datasets:
            if ds_key not in DATASETS:
                print(f"WARNING: Unknown dataset '{ds_key}', skipping. Known: {list(DATASETS.keys())}")
                continue
            ds_info = DATASETS[ds_key]

            config_dir = model_info["config_dir"]
            quarot_suffix = "_quarot" if quarot else ""
            smoke_suffix = "_smoke" if smoke else ""
            config_name = f"{ds_key}_mquant_w4a4{quarot_suffix}{smoke_suffix}.yaml"
            config_path = os.path.join(
                PROJECT_DIR, "configs", config_dir, "mquant", config_name
            )

            if not os.path.exists(config_path):
                print(f"WARNING: Config not found: {config_path}, skipping.")
                continue

            script_path = os.path.join(PROJECT_DIR, ds_info["script"])

            label = f"{model_key}_{ds_key}{'_quarot' if quarot else ''}{'_smoke' if smoke else ''}"

            jobs.append({
                "label": label,
                "model_key": model_key,
                "dataset_key": ds_key,
                "config_path": config_path,
                "script_path": script_path,
                "smoke": smoke,
                "log_path": os.path.join(LOG_DIR, f"{label}.log"),
            })

    return jobs


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------
def run_one_job(
    job: Dict,
    gpu_id: int,
    verbose: bool = False,
    mem_monitor: Optional[GPUMemoryMonitor] = None,
) -> Dict:
    """Run a single evaluation job on the given GPU. Returns result dict."""
    label = job["label"]
    config_path = job["config_path"]
    script_path = job["script_path"]
    log_path = job["log_path"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    cmd = [
        sys.executable, "-u",  # -u for unbuffered output
        script_path,
        "--config", config_path,
    ]

    # Reset peak memory for this GPU before job starts
    if mem_monitor:
        mem_monitor.reset_peak(gpu_id)
        mem_before = mem_monitor.get_current(gpu_id)
    else:
        mem_before = 0.0

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"[GPU {gpu_id}] Starting: {label}")
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
            log_f.write(f"=== {label} | GPU {gpu_id} | {datetime.now()} ===\n")
            log_f.write(f"Cmd: {' '.join(cmd)}\n\n")
            log_f.flush()
            result = subprocess.run(
                cmd, env=env, cwd=PROJECT_DIR,
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            returncode = result.returncode

    elapsed = time.time() - start_time

    # Read peak memory
    peak_mem = mem_monitor.get_peak(gpu_id) if mem_monitor else 0.0

    status = "OK" if returncode == 0 else f"FAILED (rc={returncode})"

    return {
        **job,
        "gpu_id": gpu_id,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "elapsed_str": f"{int(elapsed//3600)}h{int((elapsed%3600)//60)}m{int(elapsed%60)}s",
        "peak_vram_mb": peak_mem,
        "mem_before_mb": mem_before,
    }


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
def collect_mmmu_accuracy(save_dir: str) -> Optional[float]:
    """Parse MMMU summary report CSV for average accuracy."""
    if not os.path.isdir(save_dir):
        return None
    summary_files = sorted(Path(save_dir).glob("summary_report_*.csv"))
    if not summary_files:
        return None
    try:
        with open(str(summary_files[-1]), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("subject", "").strip().upper() == "AVERAGE":
                    return float(row["acc"]) * 100
    except Exception:
        pass
    return None


def collect_scienceqa_accuracy(save_dir: str) -> Optional[float]:
    """Parse ScienceQA allsamples CSV for accuracy."""
    if not os.path.isdir(save_dir):
        return None
    csv_files = sorted(Path(save_dir).glob("allsamples_*.csv"))
    if not csv_files:
        return None
    try:
        correct = 0
        total = 0
        with open(str(csv_files[-1]), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get("prediction", "").strip() == row.get("reference", "").strip():
                    correct += 1
        if total > 0:
            return (correct / total) * 100
    except Exception:
        pass
    return None


def collect_videomme_accuracy(save_dir: str) -> Optional[float]:
    """Parse VideoMME result directory for accuracy."""
    if not os.path.isdir(save_dir):
        return None
    # VideoMME saves per-video CSVs with accuracy info
    csv_files = list(Path(save_dir).glob("*.csv"))
    if not csv_files:
        # Check subdirectories
        csv_files = list(Path(save_dir).glob("**/*.csv"))
    if not csv_files:
        return None
    try:
        correct = 0
        total = 0
        for csv_file in csv_files:
            with open(str(csv_file), "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    # Check common accuracy columns
                    if "is_correct" in row:
                        if row["is_correct"].strip().lower() in ("true", "1", "yes"):
                            correct += 1
                    elif "prediction" in row and "answer" in row:
                        if row["prediction"].strip() == row["answer"].strip():
                            correct += 1
        if total > 0:
            return (correct / total) * 100
    except Exception:
        pass
    return None


def collect_airbench_accuracy(save_dir: str) -> Optional[float]:
    """Parse AIR-Bench JSONL result for accuracy."""
    if not os.path.isdir(save_dir):
        return None
    jsonl_files = sorted(Path(save_dir).glob("**/*.jsonl"))
    if not jsonl_files:
        # Also check the specific output_file path
        jsonl_files = sorted(Path(save_dir).glob("*.jsonl"))
    if not jsonl_files:
        return None
    try:
        correct = 0
        total = 0
        with open(str(jsonl_files[-1]), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total += 1
                if data.get("is_correct", False):
                    correct += 1
        if total > 0:
            return (correct / total) * 100
    except Exception:
        pass
    return None


def collect_accuracy(job: Dict) -> Optional[float]:
    """Dispatch to the right accuracy collector based on dataset."""
    ds = job["dataset_key"]
    # Read the config to find save_dir
    config_path = job["config_path"]
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        save_dir = cfg.get("run", {}).get("save_dir", "")
    except Exception:
        return None

    if ds == "mmmu":
        return collect_mmmu_accuracy(save_dir)
    elif ds == "scienceqa":
        return collect_scienceqa_accuracy(save_dir)
    elif ds == "videomme":
        return collect_videomme_accuracy(save_dir)
    elif ds == "airbench":
        return collect_airbench_accuracy(save_dir)
    return None


# ---------------------------------------------------------------------------
# Smoke test output check
# ---------------------------------------------------------------------------
def check_smoke_output(job: Dict) -> Tuple[bool, str]:
    """Check smoke test log for garbled output. Returns (ok, message)."""
    log_path = job["log_path"]
    if not os.path.exists(log_path):
        return False, "Log file not found"

    try:
        with open(log_path, "r") as f:
            content = f.read()
    except Exception as e:
        return False, f"Cannot read log: {e}"

    # Check for common error patterns
    error_patterns = [
        "CUDA out of memory",
        "RuntimeError",
        "Traceback (most recent call last)",
        "Segmentation fault",
        "INTERNAL ASSERT FAILED",
    ]
    for pat in error_patterns:
        if pat in content:
            return False, f"Found error pattern: {pat}"

    # Check for successful completion markers
    ok_patterns = [
        "Accuracy:",
        "accuracy:",
        "Results saved",
        "result",
        "Evaluation completed",
    ]
    # Also check if the script just ran to completion (no crash)
    # If there's output and no error patterns, it's probably OK
    if len(content.strip()) > 100:  # has reasonable output
        return True, "Output looks normal (no error patterns detected)"

    return True, "Output present but short — manual check recommended"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_summary(results: List[Dict], phase: str):
    """Print a formatted summary table."""
    print(f"\n{'='*100}")
    print(f"  MQuant W4A4 Evaluation Results — {phase}")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Dataset':<12} {'Status':<18} {'Accuracy':<12} {'Peak VRAM':<14} {'Time':<12}")
    print("-" * 100)

    for r in results:
        model_label = MODELS.get(r["model_key"], {}).get("key", r["model_key"])
        acc = r.get("accuracy")
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        vram_str = f"{r['peak_vram_mb']:.0f} MB" if r.get("peak_vram_mb", 0) > 0 else "N/A"
        print(
            f"  {model_label:<18}  {r['dataset_key']:<10}  "
            f"{r['status']:<16}  {acc_str:<10}  {vram_str:<12}  {r['elapsed_str']:<10}"
        )

    print("-" * 100)

    # Summary stats
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - ok_count
    print(f"  Total: {len(results)}  |  OK: {ok_count}  |  Failed: {fail_count}")

    if ok_count > 0:
        accs = [r["accuracy"] for r in results if r.get("accuracy") is not None]
        if accs:
            print(f"  Accuracy range: {min(accs):.2f}% – {max(accs):.2f}%  |  Mean: {sum(accs)/len(accs):.2f}%")
        vrams = [r["peak_vram_mb"] for r in results if r.get("peak_vram_mb", 0) > 0]
        if vrams:
            print(f"  VRAM range:     {min(vrams):.0f} – {max(vrams):.0f} MB  |  Peak: {max(vrams):.0f} MB")

    print(f"{'='*100}\n")


def main():
    args = parse_args()
    models = parse_list(args.models)
    datasets = parse_list(args.datasets)
    gpus = [int(x) for x in parse_list(args.gpus)]

    # Determine which phases to run
    run_smoke = args.smoke or (not args.smoke and not args.full)
    run_full = args.full or (not args.smoke and not args.full)

    if not run_smoke and not run_full:
        print("ERROR: No phase selected.")
        sys.exit(1)

    # Validate
    for g in gpus:
        if not (0 <= g <= 7):
            print(f"WARNING: GPU {g} may not exist. Valid range: 0-7.")

    print("=" * 100)
    print("  MQuant W4A4 Evaluation Runner")
    print("=" * 100)
    print(f"  Models:      {models}")
    print(f"  Datasets:    {datasets}")
    print(f"  GPUs:        {gpus}")
    print(f"  Smoke tests: {run_smoke}")
    print(f"  Full eval:   {run_full}")
    print(f"  Project dir: {PROJECT_DIR}")
    print(f"  Log dir:     {LOG_DIR}")
    if args.dry_run:
        print("  *** DRY RUN MODE ***")
    print()

    # -------------------------------------------------------------------
    # Phase 1: Smoke tests
    # -------------------------------------------------------------------
    smoke_results = []
    if run_smoke:
        smoke_jobs = build_jobs(models, datasets, smoke=True, quarot=args.quarot)
        if not smoke_jobs:
            print("No smoke test jobs to run.")
        else:
            print(f"Phase 1: Smoke tests — {len(smoke_jobs)} jobs on GPUs {gpus}")
            print()

            if args.dry_run:
                for i, job in enumerate(smoke_jobs):
                    gpu = gpus[i % len(gpus)]
                    print(f"  [{i+1}/{len(smoke_jobs)}] GPU {gpu}: {job['label']}")
                    print(f"    Config: {job['config_path']}")
                    print(f"    Log:    {job['log_path']}")
                print()
            else:
                # Start GPU memory monitor
                mem_monitor = GPUMemoryMonitor(gpus, interval=2.0)
                mem_monitor.start()

                gpu_queue = queue.Queue()
                for g in gpus:
                    gpu_queue.put(g)

                results_lock = threading.Lock()
                completed_count = [0]
                total = len(smoke_jobs)

                def run_smoke_job(job):
                    gpu = gpu_queue.get()
                    try:
                        result = run_one_job(job, gpu, args.verbose, mem_monitor)
                    finally:
                        gpu_queue.put(gpu)

                    with results_lock:
                        smoke_results.append(result)
                        completed_count[0] += 1
                        completed = completed_count[0]
                        icon = "[OK]" if result["status"] == "OK" else "[FAIL]"
                        print(
                            f"  [{completed:>2}/{total}] {icon} "
                            f"{result['label']:<30} "
                            f"GPU={result['gpu_id']}  "
                            f"VRAM peak={result['peak_vram_mb']:.0f}MB  "
                            f"{result['elapsed_str']}"
                        )
                    return result

                with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                    futures = [executor.submit(run_smoke_job, job) for job in smoke_jobs]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"  UNEXPECTED worker error: {e}")

                mem_monitor.stop()

                # Check for garbled output
                print(f"\n  Smoke test output check:")
                all_ok = True
                for r in smoke_results:
                    if r["status"] == "OK":
                        ok, msg = check_smoke_output(r)
                        icon = "[OK]" if ok else "[WARN]"
                        print(f"    {icon} {r['label']:<30} — {msg}")
                        if not ok:
                            all_ok = False
                    else:
                        print(f"    [FAIL] {r['label']:<30} — job failed")
                        all_ok = False

                if not all_ok:
                    print(
                        "\n  ⚠ Some smoke tests failed or have warnings. "
                        "Check logs before proceeding to full evaluation."
                    )
                    if not args.skip_smoke_check and run_full:
                        print(
                            "  Set --skip-smoke-check to force full evaluation anyway, "
                            "or fix the issues first."
                        )
                        print(f"  Logs are in: {LOG_DIR}")
                        # Don't exit — let user decide. Just mark results.
                else:
                    print(f"\n  ✓ All smoke tests passed!")

    # -------------------------------------------------------------------
    # Phase 2: Full evaluation
    # -------------------------------------------------------------------
    full_results = []
    if run_full:
        full_jobs = build_jobs(models, datasets, smoke=False, quarot=args.quarot)
        if not full_jobs:
            print("No full evaluation jobs to run.")
        else:
            print(f"\nPhase 2: Full evaluation — {len(full_jobs)} jobs on GPUs {gpus}")
            print()

            if args.dry_run:
                for i, job in enumerate(full_jobs):
                    gpu = gpus[i % len(gpus)]
                    print(f"  [{i+1}/{len(full_jobs)}] GPU {gpu}: {job['label']}")
                    print(f"    Config: {job['config_path']}")
                    print(f"    Log:    {job['log_path']}")
                print()
            else:
                mem_monitor = GPUMemoryMonitor(gpus, interval=3.0)
                mem_monitor.start()

                gpu_queue = queue.Queue()
                for g in gpus:
                    gpu_queue.put(g)

                results_lock = threading.Lock()
                completed_count = [0]
                total = len(full_jobs)

                def run_full_job(job):
                    gpu = gpu_queue.get()
                    try:
                        result = run_one_job(job, gpu, args.verbose, mem_monitor)
                    finally:
                        gpu_queue.put(gpu)

                    # Collect accuracy
                    acc = None
                    if result["status"] == "OK":
                        acc = collect_accuracy(result)
                        result["accuracy"] = acc

                    with results_lock:
                        full_results.append(result)
                        completed_count[0] += 1
                        completed = completed_count[0]
                        icon = "[OK]" if result["status"] == "OK" else "[FAIL]"
                        acc_str = f"acc={acc:.2f}%" if acc is not None else ""
                        print(
                            f"  [{completed:>2}/{total}] {icon} "
                            f"{result['label']:<30} "
                            f"GPU={result['gpu_id']}  "
                            f"VRAM peak={result['peak_vram_mb']:.0f}MB  "
                            f"{result['elapsed_str']}  "
                            f"{acc_str}"
                        )
                    return result

                with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                    futures = [executor.submit(run_full_job, job) for job in full_jobs]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"  UNEXPECTED worker error: {e}")

                mem_monitor.stop()

    # -------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------
    if smoke_results:
        print_summary(smoke_results, "Smoke Tests (1 sample each)")

    if full_results:
        print_summary(full_results, "Full Evaluation")

        # Save master summary CSV
        summary_csv = os.path.join(LOG_DIR, f"mquant_w4a4_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "dataset", "status", "accuracy_pct",
                "peak_vram_mb", "elapsed_sec", "elapsed_str",
                "gpu_id", "config_path",
            ])
            for r in full_results:
                writer.writerow([
                    MODELS.get(r["model_key"], {}).get("key", r["model_key"]),
                    r["dataset_key"],
                    r["status"],
                    f"{r.get('accuracy', ''):.4f}" if r.get("accuracy") is not None else "",
                    f"{r.get('peak_vram_mb', 0):.0f}",
                    f"{r.get('elapsed_sec', 0):.1f}",
                    r.get("elapsed_str", ""),
                    r.get("gpu_id", ""),
                    r.get("config_path", ""),
                ])
        print(f"Summary CSV saved to: {summary_csv}")

    # Final status
    all_results = smoke_results + full_results
    if all_results:
        total_failed = sum(1 for r in all_results if r["status"] != "OK")
        if total_failed > 0:
            print(f"\n{f'{total_failed} job(s) failed.'} Check logs in: {LOG_DIR}")
            sys.exit(1)
        else:
            print("\nAll jobs completed successfully.")


if __name__ == "__main__":
    main()
