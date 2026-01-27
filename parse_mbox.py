"""Mbox parser: stream emails, filter noise, output quarterly JSON."""

import email
import email.utils
import mailbox
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from utils import (
    clean_email_body,
    log,
    safe_json_dump,
    safe_json_load,
    strip_html_to_text,
)

MBOX_PATH = Path(__file__).parent / "src_data" / "All mail Including Spam and Trash-002.mbox"
PARSED_DIR = Path(__file__).parent / "output" / "parsed"
CHECKPOINT_PATH = Path(__file__).parent / "output" / "parsed" / "_checkpoint.json"

# Labels to exclude entirely
EXCLUDE_LABELS = {"Spam"}
# Calendar RSVP noise subjects
CALENDAR_RE = re.compile(r"^(Accepted|Declined|Tentative|Updated invitation):", re.IGNORECASE)


def _get_quarter(dt: datetime) -> str:
    """Return 'YYYY_QN' string for a datetime."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}_Q{q}"


def _parse_date(msg) -> Optional[datetime]:
    """Extract datetime from email message, return None on failure."""
    date_str = msg.get("Date", "")
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except Exception:
        return None


def _get_labels(msg) -> list[str]:
    """Extract Gmail labels from X-Gmail-Labels header."""
    raw = msg.get("X-Gmail-Labels", "")
    if not raw:
        return []
    return [l.strip() for l in raw.split(",") if l.strip()]


def _extract_body(msg) -> str:
    """Extract body text, preferring text/plain, falling back to stripped HTML."""
    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            if ct == "text/plain":
                plain_parts.append(_decode_payload(part))
            elif ct == "text/html":
                html_parts.append(_decode_payload(part))
    else:
        ct = msg.get_content_type()
        if ct == "text/plain":
            plain_parts.append(_decode_payload(msg))
        elif ct == "text/html":
            html_parts.append(_decode_payload(msg))

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return strip_html_to_text("\n".join(html_parts))
    return ""


def _decode_payload(part) -> str:
    """Decode email part payload to string with charset handling."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset)
    except (UnicodeDecodeError, LookupError):
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception:
            return payload.decode("latin-1", errors="replace")


def _should_exclude(labels: list[str], subject: str) -> tuple[bool, str]:
    """Return (should_exclude, reason) for an email."""
    label_set = set(labels)
    # Spam
    if label_set & EXCLUDE_LABELS:
        return True, "spam"
    # Promotions without Important
    if "Category Promotions" in label_set and "Important" not in label_set:
        return True, "promotions"
    # Calendar RSVP noise
    if CALENDAR_RE.match(subject or ""):
        return True, "calendar_rsvp"
    return False, ""


def parse_mbox(
    mbox_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    quarter_filter: Optional[str] = None,
) -> dict:
    """
    Parse mbox file into quarterly JSON files.

    Args:
        mbox_path: Path to mbox file. Defaults to MBOX_PATH.
        output_dir: Directory for output JSON. Defaults to PARSED_DIR.
        quarter_filter: If set, only emit this quarter (e.g. '2021_Q1').

    Returns:
        dict with parsing statistics.
    """
    mbox_path = Path(mbox_path) if mbox_path else MBOX_PATH
    output_dir = Path(output_dir) if output_dir else PARSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mbox_path.exists():
        raise FileNotFoundError(f"Mbox file not found: {mbox_path}")

    log.info(f"Opening mbox: {mbox_path}")
    log.info(f"File size: {mbox_path.stat().st_size / (1024**3):.1f} GB")

    # Load checkpoint for resume
    checkpoint = safe_json_load(str(CHECKPOINT_PATH)) or {}
    start_index = checkpoint.get("processed_count", 0)
    quarters: dict[str, list] = checkpoint.get("partial_quarters", {})
    stats = checkpoint.get("stats", {
        "total_scanned": 0,
        "excluded_spam": 0,
        "excluded_promotions": 0,
        "excluded_calendar": 0,
        "no_date": 0,
        "kept": 0,
    })

    if start_index > 0:
        log.info(f"Resuming from checkpoint: {start_index} emails already processed")

    mbox = mailbox.mbox(str(mbox_path))
    total_estimate = 49_308  # from plan

    with tqdm(total=total_estimate, initial=start_index, desc="Parsing emails") as pbar:
        for i, msg in enumerate(mbox):
            if i < start_index:
                continue

            stats["total_scanned"] = i + 1

            # Parse date
            dt = _parse_date(msg)
            if dt is None:
                stats["no_date"] += 1
                pbar.update(1)
                continue

            quarter = _get_quarter(dt)

            # Quarter filter
            if quarter_filter and quarter != quarter_filter:
                pbar.update(1)
                continue

            # Extract metadata
            subject = msg.get("Subject", "") or ""
            labels = _get_labels(msg)

            # Filter
            excluded, reason = _should_exclude(labels, subject)
            if excluded:
                stats[f"excluded_{reason}"] = stats.get(f"excluded_{reason}", 0) + 1
                pbar.update(1)
                continue

            # Extract fields
            from_addr = msg.get("From", "") or ""
            to_addr = msg.get("To", "") or ""
            cc_addr = msg.get("Cc", "") or ""
            message_id = msg.get("Message-ID", "") or ""
            thread_id = msg.get("X-GM-THRID", "") or ""
            in_reply_to = msg.get("In-Reply-To", "") or ""

            is_sent = any(
                lbl in ("Sent", "SENT") for lbl in labels
            ) or "clay" in from_addr.lower()

            body = _extract_body(msg)
            body = clean_email_body(body)

            record = {
                "message_id": message_id,
                "thread_id": thread_id,
                "in_reply_to": in_reply_to,
                "date": dt.isoformat(),
                "from": from_addr,
                "to": to_addr,
                "cc": cc_addr,
                "subject": subject,
                "labels": labels,
                "is_sent": is_sent,
                "body": body,
            }

            if quarter not in quarters:
                quarters[quarter] = []
            quarters[quarter].append(record)
            stats["kept"] += 1

            # Checkpoint every 1000 emails
            if (i + 1) % 1000 == 0:
                pbar.set_postfix(kept=stats["kept"], quarters=len(quarters))
                _save_checkpoint(i + 1, quarters, stats)

            pbar.update(1)

    mbox.close()

    # Write quarterly JSON files
    log.info(f"Writing {len(quarters)} quarterly files...")
    for quarter, emails in sorted(quarters.items()):
        # Sort emails within quarter by date
        emails.sort(key=lambda e: e["date"])
        quarter_data = {
            "quarter": quarter,
            "email_count": len(emails),
            "date_range": {
                "start": emails[0]["date"] if emails else None,
                "end": emails[-1]["date"] if emails else None,
            },
            "emails": emails,
        }
        outpath = output_dir / f"{quarter}.json"
        safe_json_dump(quarter_data, str(outpath))
        log.info(f"  {quarter}: {len(emails)} emails -> {outpath.name}")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        os.unlink(str(CHECKPOINT_PATH))

    stats["quarters_written"] = len(quarters)
    stats["quarter_counts"] = {q: len(e) for q, e in sorted(quarters.items())}

    log.info(f"Parsing complete. {stats['kept']} emails kept across {len(quarters)} quarters.")
    log.info(f"Excluded: spam={stats.get('excluded_spam', 0)}, "
             f"promotions={stats.get('excluded_promotions', 0)}, "
             f"calendar={stats.get('excluded_calendar', 0)}, "
             f"no_date={stats['no_date']}")

    return stats


def _save_checkpoint(processed_count: int, quarters: dict, stats: dict) -> None:
    """Save progress checkpoint (without full email data — just counts)."""
    # We save the full quarters data so we can resume
    # This could be large, but it's the only way to truly resume
    checkpoint = {
        "processed_count": processed_count,
        "stats": stats,
        "partial_quarters": quarters,
    }
    safe_json_dump(checkpoint, str(CHECKPOINT_PATH))


def get_parsed_quarters(parsed_dir: Optional[str] = None) -> list[str]:
    """Return sorted list of quarter names that have been parsed."""
    d = Path(parsed_dir) if parsed_dir else PARSED_DIR
    quarters = []
    for f in d.glob("*.json"):
        if f.stem.startswith("_"):
            continue
        quarters.append(f.stem)
    return sorted(quarters)


def load_quarter(quarter: str, parsed_dir: Optional[str] = None) -> Optional[dict]:
    """Load a parsed quarter JSON file."""
    d = Path(parsed_dir) if parsed_dir else PARSED_DIR
    return safe_json_load(str(d / f"{quarter}.json"))


if __name__ == "__main__":
    stats = parse_mbox()
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    import json
