#!/usr/bin/env python3
"""结果文件 -> accuracy。

支持 4 种 evaluator 的输出，按列名自动识别:
  ScienceQA  CSV   : pid, question, reference, prediction, raw_output
  MMMU       CSV   : id, subject, question, answer, prediction, is_correct, raw_output
  Video-MME  CSV   : video_id, question, answer, prediction, raw_response, correct
  AIR-Bench  JSONL : answer_gt, response, task_name, dataset_name, ...

用法::

    python scripts/score_results.py <path> [<path> ...]
    python scripts/score_results.py <dir>            # 递归找 *.csv / *.jsonl
    python scripts/score_results.py <path> --by subject     # 分组明细

评测还在跑时也能用 —— 结果文件每 10 个样本 flush 一次，
所以这就是"当前 running accuracy"。
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

# evaluator 里 prediction 失败时写入的占位符
_FAILED = {"错误", "Failed", "failed", ""}

# (真值列, 预测列, 预置的正误列 或 None, 默认分组列 或 None)
_CSV_SCHEMAS = [
    ("reference", "prediction", None, None),          # ScienceQA
    ("answer", "prediction", "is_correct", "subject"),  # MMMU
    ("answer", "prediction", "correct", None),         # Video-MME
]


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _pick_csv_schema(fieldnames):
    names = set(fieldnames or ())
    for gt, pred, flag, group in _CSV_SCHEMAS:
        if gt in names and pred in names:
            # 只在列真实存在时才使用预置正误列 / 分组列
            return gt, pred, (flag if flag in names else None), (group if group in names else None)
    return None


def _load_csv(path):
    """返回 (rows, group_key)，row 为 (gt, pred, correct_or_None, group_or_None)。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        schema = _pick_csv_schema(reader.fieldnames)
        if schema is None:
            raise ValueError(
                f"{path}: unrecognized columns {reader.fieldnames}. "
                f"Expected one of the ScienceQA / MMMU / Video-MME schemas."
            )
        gt_k, pred_k, flag_k, group_k = schema
        rows = []
        for r in reader:
            gt = str(r.get(gt_k, "")).strip()
            pred = str(r.get(pred_k, "")).strip()
            flag = _truthy(r.get(flag_k)) if flag_k else None
            group = r.get(group_k) if group_k else None
            rows.append((gt, pred, flag, group))
    return rows, group_k


def _load_jsonl(path):
    """AIR-Bench: answer_gt vs response。分组用 task_name。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # 评测中途被打断时最后一行可能不完整，跳过而不是崩掉
                print(f"  warn: {path}:{line_no} is not valid JSON, skipped", file=sys.stderr)
                continue
            gt = str(d.get("answer_gt", "")).strip()
            pred = str(d.get("response", "")).strip()
            rows.append((gt, pred, None, d.get("task_name")))
    return rows, "task_name"


def score(path):
    if path.lower().endswith(".jsonl"):
        return _load_jsonl(path)
    return _load_csv(path)


def summarize(rows, group_by=None):
    total = len(rows)
    correct = 0
    failed = 0
    groups = {}

    for gt, pred, flag, group in rows:
        # 优先用 evaluator 已经算好的正误列，避免大小写/规范化口径不一致
        ok = flag if flag is not None else (gt == pred and pred not in _FAILED)
        if pred in _FAILED:
            failed += 1
        if ok:
            correct += 1
        if group_by:
            g = groups.setdefault(group or "(none)", [0, 0])
            g[1] += 1
            if ok:
                g[0] += 1

    return total, correct, failed, groups


def _fmt(correct, total):
    if total == 0:
        return "n/a (0 samples)"
    return f"{correct / total:.4f} ({correct}/{total}, {correct / total * 100:.2f}%)"


def collect_paths(targets):
    out = []
    for t in targets:
        if os.path.isdir(t):
            for root, _, files in os.walk(t):
                for fn in sorted(files):
                    if fn.endswith((".csv", ".jsonl")):
                        out.append(os.path.join(root, fn))
        else:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute accuracy from MorphoQuant result files.")
    ap.add_argument("paths", nargs="+", help="result file(s) or directory to scan")
    ap.add_argument("--by", metavar="COL", default=None,
                    help="group breakdown; use the schema's group column (e.g. subject / task_name), "
                         "or 'auto' to use the default for that format")
    args = ap.parse_args()

    paths = collect_paths(args.paths)
    if not paths:
        print("No .csv / .jsonl files found.", file=sys.stderr)
        return 1

    exit_code = 0
    for path in paths:
        print(f"\n=== {path}")
        try:
            rows, default_group = score(path)
        except (OSError, ValueError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            exit_code = 1
            continue

        group_by = default_group if args.by == "auto" else args.by
        if group_by and not default_group:
            print(f"  note: this format has no group column; --by ignored")
            group_by = None

        total, correct, failed, groups = summarize(rows, group_by)
        print(f"  Accuracy : {_fmt(correct, total)}")
        if failed:
            print(f"  Failed   : {failed} ({failed / total * 100:.2f}%)  <- 推理报错/无法解析的样本")

        if groups:
            print(f"  --- by {group_by} ---")
            for name, (c, t) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
                print(f"    {name:<40} {_fmt(c, t)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
