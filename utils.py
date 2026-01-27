"""Shared utilities for mbox email archive analyzer."""

import json
import logging
import os
import re
import tempfile
from pathlib import Path

try:
    import html2text
    _H2T = html2text.HTML2Text()
    _H2T.ignore_links = False
    _H2T.ignore_images = True
    _H2T.body_width = 0
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project-wide logger."""
    logger = logging.getLogger("mbox_analyzer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Fallback HTML-to-text when html2text is unavailable."""

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self.result.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.result.append(data)

    def get_text(self) -> str:
        return "".join(self.result)


def strip_html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    if not html:
        return ""
    if HAS_HTML2TEXT:
        return _H2T.handle(html)
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~3 chars per token, conservative)."""
    if not text:
        return 0
    return len(text) // 3


# ---------------------------------------------------------------------------
# Body processing
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    # Slack footers
    re.compile(r"(?:^|\n)(?:Reply to this email directly.*|---\nSent via Slack.*|Join the conversation on Slack.*)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\n-{2,}\nSent from Slack[\s\S]*$", re.IGNORECASE),
    # Jira / Confluence footers
    re.compile(r"\n-{2,}\nThis message was sent by Atlassian[\s\S]*$", re.IGNORECASE),
    re.compile(r"\nView (?:issue|page|comment) in (?:Jira|Confluence):\s*https?://\S+", re.IGNORECASE),
    # GitHub footers
    re.compile(r"\n-{2,}\nYou are receiving this because[\s\S]*$", re.IGNORECASE),
    re.compile(r"\nReply to this email directly or view it on GitHub[\s\S]*$", re.IGNORECASE),
    # Generic email signatures (heuristic: "-- " on its own line)
    re.compile(r"\n-- \n[\s\S]{0,500}$"),
    # Tracking URLs / pixel lines
    re.compile(r"\nhttps?://\S{200,}\s*$"),
    # Confidentiality disclaimers
    re.compile(r"\n(?:CONFIDENTIAL|DISCLAIMER|This email and any attachments)[\s\S]{0,1000}$", re.IGNORECASE),
    # Calendar invite boilerplate
    re.compile(r"\nmore details\s*>>?\s*https?://\S+", re.IGNORECASE),
    re.compile(r"\nInvitation from Google Calendar[\s\S]*$", re.IGNORECASE),
]


def clean_email_body(text: str) -> str:
    """Remove common boilerplate from email body text."""
    if not text:
        return ""
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def truncate_body(text: str, max_chars: int = 2000) -> str:
    """Truncate body at sentence boundary, capped at max_chars."""
    if not text or len(text) <= max_chars:
        return text or ""
    truncated = text[:max_chars]
    # Try to break at last sentence boundary
    for sep in (". ", ".\n", "!\n", "?\n", "! ", "? ", "\n\n"):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + len(sep)].rstrip() + " [truncated]"
    return truncated.rstrip() + " [truncated]"


# ---------------------------------------------------------------------------
# Safe JSON I/O
# ---------------------------------------------------------------------------

def safe_json_dump(data, filepath: str) -> None:
    """Atomically write JSON: write to temp file, then rename."""
    filepath = str(filepath)
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, filepath)
    except Exception:
        os.unlink(tmp)
        raise


def safe_json_load(filepath: str):
    """Load JSON from file, return None if missing or corrupt."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
