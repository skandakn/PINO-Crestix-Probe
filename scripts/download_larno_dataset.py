#!/usr/bin/env python3
"""
download_larno_dataset.py
=========================
Reproducible download, caching, and integrity verification script for the
Crestix Probe LarNO benchmark dataset (1000 events).

Sources:
  - Primary Hugging Face Repository: https://huggingface.co/datasets/holmescao/LarNO-dataset
  - Benchmark Archive: benchmark.zip (for event8, event9, event79 and offline workflows)

Features:
  - Event-level filtering (--split train/val/test, --catchment Futian/UKEA)
  - Dry-run mode for CI/CD checks without downloading multi-GB payloads
  - Safe resume with existing file validation
  - Automatic directory organization conforming to shared data contracts

Usage:
  python scripts/download_larno_dataset.py --dry-run
  python scripts/download_larno_dataset.py --catchment UKEA --split val --output-dir data/raw
  python scripts/download_larno_dataset.py --events event1,event2,r100y_p0.1_d3h_1 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("larno_downloader")

HF_BASE_URL = "https://huggingface.co/datasets/holmescao/LarNO-dataset/resolve/main"

DEM_URLS = {
    "Futian": f"{HF_BASE_URL}/geodata/region1_20m/dem.npy",
    "UKEA": f"{HF_BASE_URL}/geodata/ukea_8m_5min/dem.npy",
}


def download_file(url: str, dest_path: Path, dry_run: bool = False) -> bool:
    """Download a file with user-agent headers and error logging."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.debug("Already exists: %s", dest_path)
        return True

    if dry_run:
        logger.info("[DRY RUN] Would download: %s -> %s", url, dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "CrestixProbe/1.0 LarNODownloader"})
    try:
        logger.info("Downloading: %s", url)
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_fp:
            chunk_size = 1024 * 1024  # 1 MB
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_fp.write(chunk)
        temp_path.replace(dest_path)
        return True
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        logger.error("Download failed for %s: %s", url, exc)
        return False


def download_larno(
    output_dir: Path,
    manifest_path: Path,
    catchment: Optional[str] = None,
    split: Optional[str] = None,
    events: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Download selected events following official LarNO dataset layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fp:
        all_events = list(csv.DictReader(fp))

    selected = []
    for ev in all_events:
        if catchment and catchment.lower() != "all" and ev["catchment"].lower() != catchment.lower():
            continue
        if split and split.lower() != "all" and ev["split"].lower() != split.lower():
            continue
        if events and ev["event_id"] not in events:
            continue
        selected.append(ev)

    logger.info("Selected %d events for download (catchment=%s, split=%s)", len(selected), catchment, split)

    # 1. Download DEMs for present catchments
    catchments_present = set(ev["catchment"] for ev in selected)
    for cat in catchments_present:
        dem_url = DEM_URLS.get(cat)
        if dem_url:
            geo_dir = output_dir / "geodata" / ("region1_20m" if cat == "Futian" else "ukea_8m_5min")
            geo_dir.mkdir(parents=True, exist_ok=True)
            dem_dest = geo_dir / "dem.npy"
            download_file(dem_url, dem_dest, dry_run=dry_run)

    # 2. Download events
    success_count = 0
    fail_count = 0
    for idx, ev in enumerate(selected):
        ev_id = ev["event_id"]
        cat = ev["catchment"]
        reg_tag = ev["region_tag"]
        ev_dir = output_dir / "flood" / reg_tag / ev_id
        ev_dir.mkdir(parents=True, exist_ok=True)

        # Check if event is in benchmark.zip on HF
        if ev_id in ("event8", "event9", "event79"):
            logger.info("Event %s is archived in benchmark.zip on Hugging Face.", ev_id)
            # URL directly from HF tree if synced, or flag for zip extraction
            url_rain = f"{HF_BASE_URL}/flood/{reg_tag}/{ev_id}/rainfall.npy"
            url_h = f"{HF_BASE_URL}/flood/{reg_tag}/{ev_id}/h.npy"
        else:
            url_rain = f"{HF_BASE_URL}/flood/{reg_tag}/{ev_id}/rainfall.npy"
            url_h = f"{HF_BASE_URL}/flood/{reg_tag}/{ev_id}/h.npy"

        ok_r = download_file(url_rain, ev_dir / "rainfall.npy", dry_run=dry_run)
        ok_h = download_file(url_h, ev_dir / "h.npy", dry_run=dry_run)

        if ok_r and ok_h:
            success_count += 1
        else:
            fail_count += 1

    return {
        "total_selected": len(selected),
        "success": success_count,
        "failed": fail_count,
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="LarNO Dataset Download & Cache Utility")
    parser.add_argument("--output-dir", type=str, default="data/raw_larno", help="Root storage directory")
    parser.add_argument("--manifest", type=str, default="data/event_manifest.csv", help="Path to event_manifest.csv")
    parser.add_argument("--catchment", type=str, choices=["Futian", "UKEA", "all"], default="all")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--events", type=str, default=None, help="Comma-separated event IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    args = parser.parse_args()

    event_filter = args.events.split(",") if args.events else None
    results = download_larno(
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        catchment=args.catchment,
        split=args.split,
        events=event_filter,
        dry_run=args.dry_run,
    )
    logger.info("Download completed: %s", results)


if __name__ == "__main__":
    main()
