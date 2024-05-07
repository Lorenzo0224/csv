"""Misc utils."""
import logging
from pathlib import Path
from typing import List
import re

from rich.logging import RichHandler


def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    handlers = [logging.FileHandler(log_path), RichHandler(rich_tracebacks=True)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        print("111", label, pred)
        mets["total"] += 1
        if task in {
            "data_imputation",
        }:
            crc = pred == label
        elif task in {"entity_matching", "schema_matching", "error_detection_spelling"}:
            crc = pred.startswith(label)
        elif task in {"error_detection"}:
            pred = pred.split("\n\n")[-1]
            breakpoint()
            crc = pred.endswith(label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1


def compute_metrics_for_sample_skipping(preds: List, golds: List, task: str, skippable: List):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    true_pred = []
    idx = 0
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        print("111", label, pred)
        mets["total"] += 1
        if task in {
            "data_imputation",
        }:
            crc = pred == label
        elif task in {"entity_matching", "schema_matching", "error_detection_spelling"}:
            crc = pred.startswith(label)
        elif task in {"error_detection"}:
            pred = pred.split("\n\n")[-1]
            breakpoint()
            crc = pred.endswith(label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc or idx in skippable:
            mets["crc"] += 1
            true_pred.append(idx)
        if label == "yes":
            if crc or idx in skippable:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc or idx in skippable:
                mets["tn"] += 1
            else:
                mets["fp"] += 1
        idx += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1, true_pred





def seperate_k_chatgpt(input_string, k):
    print("the ans", input_string)
    #matches = re.search(pattern, input_string)

    matchesYN = re.findall(r'(?:Yes|No)', input_string)
    ans = []

    for match in matchesYN:
        ans.append(match)
    while(len(ans)<k):
        ans.append('')
    return ans[:k]

def seperate_k(input_string, k):
    print("the ans", input_string)
    #matches = re.search(pattern, input_string)
    matches = re.findall(r'\bA\[(\d+)\]:(\S+)', input_string)
    matchesQ = re.findall(r'\bQ\[(\d+)\]:(\S+)', input_string)
    matchesYN = re.findall(r'(?:Yes|No)', input_string)
    ans = []
    if(len(matchesQ)>0):
        for match in matchesQ:
            ans.append(match[1])
    elif(len(matches)>0):
        for match in matches:
            ans.append(match[1])
    else:
        for match in matchesYN:
            ans.append(match)
    while(len(ans)<k):
        ans.append('')
    return ans[:k]
'''
    values = []
    if matches:
        for i in range(1, k + 1):
            value = matches.group(i)
            values.append(value)
            print(f"A{i} value:", value)
    else:
        print("No match found.")'''

def compute_bp_metrics(preds, List2, str, num):
    """Compute metrics."""
    try:
        list_of_p = []
        for pred in preds:
            for p in pred:
                list_of_p.append(p)
        compute_metrics(list_of_p, List2, "entity_matching")
    except Exception as e:
        list_of_p = []
    return list_of_p
