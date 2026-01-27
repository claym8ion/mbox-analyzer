#!/usr/bin/env python3
"""Mbox Email Archive Analyzer — Pipeline Orchestrator.

Usage:
    python main.py                      # Full pipeline
    python main.py --dry-run            # Parse + cost estimate, no API calls
    python main.py --quarter 2021_Q1    # Single quarter only
    python main.py --skip-parse         # Reuse existing parsed files
    python main.py --skip-analyze       # Reuse existing analysis, regenerate report
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from utils import log, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Mbox Email Archive Analyzer — parse, analyze, report"
    )
    parser.add_argument(
        "--skip-parse", action="store_true",
        help="Skip parsing; reuse existing quarterly JSON files",
    )
    parser.add_argument(
        "--skip-analyze", action="store_true",
        help="Skip analysis; reuse existing analysis files",
    )
    parser.add_argument(
        "--quarter", type=str, default=None,
        help="Process a single quarter, e.g. 2021_Q1",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse + estimate cost only; no API calls",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--batch-delay", type=float, default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    log.info("=" * 60)
    log.info("Mbox Email Archive Analyzer")
    log.info("=" * 60)

    # Validate API key (unless dry-run or skip-analyze)
    if not args.dry_run and not args.skip_analyze:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            log.error("ANTHROPIC_API_KEY environment variable not set.")
            log.error("Set it with: export ANTHROPIC_API_KEY=sk-...")
            sys.exit(1)
        log.info("API key found.")

    parse_stats = None
    analysis_stats = None

    # --- Step 1: Parse ---
    if not args.skip_parse:
        log.info("-" * 40)
        log.info("STEP 1: Parsing mbox file")
        log.info("-" * 40)

        from parse_mbox import parse_mbox
        parse_stats = parse_mbox(quarter_filter=args.quarter)
        log.info(f"Parse stats: {json.dumps(parse_stats, indent=2, default=str)}")
    else:
        log.info("Skipping parse (--skip-parse)")

    # --- Step 2: Analyze ---
    if not args.skip_analyze:
        log.info("-" * 40)
        log.info("STEP 2: Analyzing with Claude API")
        log.info("-" * 40)

        from analyze import run_analysis
        analysis_stats = run_analysis(
            model=args.model,
            quarter_filter=args.quarter,
            batch_delay=args.batch_delay,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            log.info("DRY RUN — Cost estimate:")
            log.info(json.dumps(analysis_stats, indent=2, default=str))
            log.info("No API calls were made.")
            return
        else:
            log.info(f"Analysis stats: {json.dumps(analysis_stats, indent=2, default=str)}")
    else:
        log.info("Skipping analysis (--skip-analyze)")

    # --- Step 3: Generate Report ---
    log.info("-" * 40)
    log.info("STEP 3: Generating report")
    log.info("-" * 40)

    from report import generate_report
    report_path = generate_report(
        parse_stats=parse_stats,
        analysis_stats=analysis_stats,
    )

    if report_path:
        log.info(f"Report generated: {report_path}")
    else:
        log.error("Report generation failed.")

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
