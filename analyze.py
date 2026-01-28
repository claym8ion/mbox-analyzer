"""Claude API analysis: two-pass analysis of parsed email data."""

import json
import time
from pathlib import Path
from typing import Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from parse_mbox import get_parsed_quarters, load_quarter
from utils import (
    estimate_tokens,
    log,
    safe_json_dump,
    safe_json_load,
    truncate_body,
)

ANALYSIS_DIR = Path(__file__).parent / "output" / "analysis"
SUMMARY_DIR = Path(__file__).parent / "output" / "summaries"
PROGRESS_PATH = Path(__file__).parent / "output" / "analysis" / "_progress.json"

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS_PER_BATCH = 100_000  # target input tokens per API call (conservative for 200K context)
BATCH_OUTPUT_TOKENS = 8192


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Return True for rate-limit, server errors, and connection errors."""
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    if isinstance(exc, (anthropic.APIConnectionError, ConnectionError)):
        return True
    return False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PASS1_SYSTEM = """You are an expert career analyst reviewing email archives to build a comprehensive professional profile.
You will receive a batch of emails from a single quarter. Analyze them for:

1. **Projects**: Named initiatives, products, features, or workstreams. Include role, collaborators, and status.
2. **Accomplishments**: Concrete achievements, launches, completions, positive outcomes. Be specific.
3. **Feedback & Recognition**: Praise received, constructive feedback received, feedback given to others. Include who said what.
4. **Key Relationships**: Who does this person collaborate with most? Who are their stakeholders?
5. **Themes**: Recurring topics, technologies, domains of work.

Focus on the email sender/owner "Clay" — this is their email archive. Identify what Clay did, led, contributed to, and how others perceived their work.

Return valid JSON with this structure:
{
  "quarter": "<quarter>",
  "projects": [
    {"name": "", "description": "", "role": "", "collaborators": [], "status": "", "evidence": ""}
  ],
  "accomplishments": [
    {"description": "", "category": "", "evidence": "", "date": ""}
  ],
  "feedback": {
    "received_positive": [{"from": "", "content": "", "context": "", "date": ""}],
    "received_constructive": [{"from": "", "content": "", "context": "", "date": ""}],
    "given_to_others": [{"to": "", "content": "", "context": "", "date": ""}]
  },
  "key_relationships": [
    {"person": "", "relationship": "", "context": ""}
  ],
  "themes": [""]
}"""

PASS1_USER_TEMPLATE = """Quarter: {quarter} (batch {batch_num}/{total_batches})

Below are {email_count} emails from this quarter. Analyze them for projects, accomplishments, feedback, and relationships involving Clay.

--- EMAILS ---
{emails_text}
--- END EMAILS ---

Return your analysis as valid JSON matching the specified schema."""


PASS2_SYSTEM = """You are an expert career analyst performing a final consolidation of quarterly email analyses spanning multiple years at a company.

Your task:
1. **Deduplicate projects** that appear across quarters — merge into unified project entries with timelines.
2. **Build a career arc** — how did Clay's role, scope, and impact evolve over time?
3. **Synthesize themes** — what are the major recurring themes across the full tenure?
4. **Rank accomplishments** — identify the top 10-15 most significant accomplishments.
5. **Consolidate feedback** — identify patterns in feedback received and given.
6. **Map key relationships** — who were the most important professional relationships?

Return valid JSON:
{
  "projects": [
    {"name": "", "description": "", "role": "", "timeline": "", "collaborators": [], "key_outcomes": [], "status": ""}
  ],
  "top_accomplishments": [
    {"description": "", "category": "", "quarter": "", "impact": ""}
  ],
  "feedback_summary": {
    "strengths_identified": [""],
    "growth_areas": [""],
    "notable_positive": [{"from": "", "content": "", "context": ""}],
    "notable_constructive": [{"from": "", "content": "", "context": ""}],
    "feedback_given_themes": [""]
  },
  "key_relationships": [
    {"person": "", "relationship": "", "significance": ""}
  ],
  "career_arc": {
    "phases": [
      {"period": "", "title_or_role": "", "focus": "", "key_shifts": ""}
    ],
    "overall_trajectory": ""
  },
  "themes": [""],
  "executive_summary": ""
}"""

PASS2_USER_TEMPLATE = """Below are quarterly analyses spanning Clay's tenure. Consolidate them into a unified career profile.

{quarterly_summaries}

Return your consolidated analysis as valid JSON matching the specified schema."""


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

def _format_email_compact(email_rec: dict) -> str:
    """Format a single email record into compact text for the prompt."""
    parts = []
    parts.append(f"Date: {email_rec['date']}")
    parts.append(f"From: {email_rec['from']}")
    if email_rec.get("to"):
        parts.append(f"To: {email_rec['to']}")
    parts.append(f"Subject: {email_rec['subject']}")
    if email_rec.get("labels"):
        parts.append(f"Labels: {', '.join(email_rec['labels'])}")
    body = truncate_body(email_rec.get("body", ""))
    if body:
        parts.append(f"Body:\n{body}")
    return "\n".join(parts)


def _batch_emails(emails: list[dict], max_tokens: int = MAX_TOKENS_PER_BATCH) -> list[list[dict]]:
    """Split emails into batches that fit within token budget."""
    # Group by thread first for coherence
    threads: dict[str, list[dict]] = {}
    no_thread = []
    for em in emails:
        tid = em.get("thread_id", "")
        if tid:
            threads.setdefault(tid, []).append(em)
        else:
            no_thread.append(em)

    # Flatten thread groups back into ordered list (threads stay together)
    ordered = []
    for tid_emails in threads.values():
        ordered.extend(tid_emails)
    ordered.extend(no_thread)

    batches = []
    current_batch = []
    current_tokens = 0
    # Reserve tokens for system prompt + framing
    overhead = estimate_tokens(PASS1_SYSTEM) + 500

    for em in ordered:
        formatted = _format_email_compact(em)
        em_tokens = estimate_tokens(formatted)
        if current_batch and (current_tokens + em_tokens + overhead) > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(em)
        current_tokens += em_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

RATE_LIMIT_TPM = 30_000  # tokens per minute for the org


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=10, min=30, max=300),
    stop=stop_after_attempt(10),
    before_sleep=lambda rs: log.warning(
        f"Retrying API call (attempt {rs.attempt_number}) after error: {rs.outcome.exception()}"
    ),
)
def _call_claude(
    client: anthropic.Anthropic,
    system: str,
    user_msg: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = BATCH_OUTPUT_TOKENS,
) -> str:
    """Make a single Claude API call with retry logic."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def _parse_json_response(text: str) -> dict:
    """Parse JSON from Claude's response, handling markdown code fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON response: {e}")
        log.error(f"Response text (first 500 chars): {text[:500]}")
        # Return wrapped raw text so we don't lose data
        return {"raw_response": text, "parse_error": str(e)}


# ---------------------------------------------------------------------------
# Pass 1: Per-quarter analysis
# ---------------------------------------------------------------------------

def analyze_quarter(
    client: anthropic.Anthropic,
    quarter: str,
    model: str = DEFAULT_MODEL,
    batch_delay: float = 1.0,
    dry_run: bool = False,
    parsed_dir: Optional[str] = None,
) -> dict:
    """Run Pass 1 analysis on a single quarter."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress = safe_json_load(str(PROGRESS_PATH)) or {}
    quarter_progress = progress.get(quarter, {})

    # Check if already complete
    result_path = ANALYSIS_DIR / f"{quarter}_analysis.json"
    existing = safe_json_load(str(result_path))
    if existing and not existing.get("parse_error"):
        log.info(f"Quarter {quarter} already analyzed, skipping.")
        return existing

    # Load parsed data
    quarter_data = load_quarter(quarter, parsed_dir)
    if not quarter_data:
        log.warning(f"No parsed data for {quarter}")
        return {}

    emails = quarter_data["emails"]
    batches = _batch_emails(emails)
    log.info(f"Quarter {quarter}: {len(emails)} emails in {len(batches)} batches")

    if dry_run:
        total_tokens = 0
        for batch in batches:
            text = "\n\n===\n\n".join(_format_email_compact(e) for e in batch)
            total_tokens += estimate_tokens(PASS1_SYSTEM + text)
        return {
            "quarter": quarter,
            "email_count": len(emails),
            "batch_count": len(batches),
            "estimated_input_tokens": total_tokens,
            "estimated_cost_usd": round(total_tokens * 3 / 1_000_000, 2),
            "dry_run": True,
        }

    # Process batches
    batch_results = []
    completed_batches = quarter_progress.get("completed_batches", [])

    for batch_idx, batch in enumerate(batches):
        batch_key = f"{quarter}_batch_{batch_idx}"

        # Check if batch already done
        if batch_key in completed_batches:
            cached = safe_json_load(str(ANALYSIS_DIR / f"{batch_key}.json"))
            if cached:
                batch_results.append(cached)
                log.info(f"  Batch {batch_idx + 1}/{len(batches)} — cached")
                continue

        emails_text = "\n\n===\n\n".join(_format_email_compact(e) for e in batch)
        user_msg = PASS1_USER_TEMPLATE.format(
            quarter=quarter,
            batch_num=batch_idx + 1,
            total_batches=len(batches),
            email_count=len(batch),
            emails_text=emails_text,
        )

        input_tokens = estimate_tokens(PASS1_SYSTEM + user_msg)
        log.info(f"  Batch {batch_idx + 1}/{len(batches)} — {len(batch)} emails, "
                 f"~{input_tokens}tok input")

        raw = _call_claude(client, PASS1_SYSTEM, user_msg, model=model)
        result = _parse_json_response(raw)

        # Save batch result
        safe_json_dump(result, str(ANALYSIS_DIR / f"{batch_key}.json"))
        batch_results.append(result)

        # Update progress
        completed_batches.append(batch_key)
        progress[quarter] = {"completed_batches": completed_batches}
        safe_json_dump(progress, str(PROGRESS_PATH))

        if batch_idx < len(batches) - 1:
            # Rate-limit-aware delay: wait long enough for token budget to refill
            rate_delay = (input_tokens / RATE_LIMIT_TPM) * 60 + 5  # seconds + buffer
            actual_delay = max(batch_delay, rate_delay)
            log.info(f"  Waiting {actual_delay:.0f}s for rate limit budget...")
            time.sleep(actual_delay)

    # Merge batch results into quarter analysis
    merged = _merge_batch_results(quarter, batch_results)
    safe_json_dump(merged, str(result_path))

    # Mark quarter as complete in progress
    progress[quarter]["complete"] = True
    safe_json_dump(progress, str(PROGRESS_PATH))

    log.info(f"Quarter {quarter} analysis complete: {len(merged.get('projects', []))} projects, "
             f"{len(merged.get('accomplishments', []))} accomplishments")

    return merged


def _merge_batch_results(quarter: str, batch_results: list[dict]) -> dict:
    """Merge multiple batch analysis results into a single quarter result."""
    merged = {
        "quarter": quarter,
        "projects": [],
        "accomplishments": [],
        "feedback": {
            "received_positive": [],
            "received_constructive": [],
            "given_to_others": [],
        },
        "key_relationships": [],
        "themes": [],
    }

    seen_project_names = set()
    for result in batch_results:
        if "raw_response" in result:
            continue

        for p in result.get("projects", []):
            name = p.get("name", "").lower().strip()
            if name and name not in seen_project_names:
                seen_project_names.add(name)
                merged["projects"].append(p)

        merged["accomplishments"].extend(result.get("accomplishments", []))

        fb = result.get("feedback", {})
        if isinstance(fb, dict):
            merged["feedback"]["received_positive"].extend(fb.get("received_positive", []))
            merged["feedback"]["received_constructive"].extend(fb.get("received_constructive", []))
            merged["feedback"]["given_to_others"].extend(fb.get("given_to_others", []))

        merged["key_relationships"].extend(result.get("key_relationships", []))

        for theme in result.get("themes", []):
            if theme not in merged["themes"]:
                merged["themes"].append(theme)

    return merged


# ---------------------------------------------------------------------------
# Pass 2: Consolidation
# ---------------------------------------------------------------------------

def _compact_quarter_summary(analysis: dict) -> dict:
    """Create a compact summary of a quarter analysis for consolidation."""
    return {
        "quarter": analysis.get("quarter", ""),
        "projects": [
            {"name": p.get("name", ""), "role": p.get("role", ""), "status": p.get("status", "")}
            for p in analysis.get("projects", [])[:15]  # top 15 projects
        ],
        "accomplishments": [
            {"description": a.get("description", ""), "category": a.get("category", "")}
            for a in analysis.get("accomplishments", [])[:10]  # top 10
        ],
        "feedback_positive": [
            f.get("content", "")[:200] for f in analysis.get("feedback", {}).get("received_positive", [])[:5]
        ],
        "feedback_constructive": [
            f.get("content", "")[:200] for f in analysis.get("feedback", {}).get("received_constructive", [])[:3]
        ],
        "key_relationships": [
            r.get("person", "") for r in analysis.get("key_relationships", [])[:10]
        ],
        "themes": analysis.get("themes", [])[:10],
    }


YEAR_CONSOLIDATION_SYSTEM = """You are an expert career analyst consolidating quarterly email analyses for a single year.

Synthesize the quarters into a yearly summary with:
1. **Projects**: Major initiatives that year (deduplicate across quarters)
2. **Accomplishments**: Top 10 achievements for the year
3. **Feedback**: Key positive and constructive feedback received
4. **Relationships**: Most important professional relationships
5. **Themes**: Dominant themes for the year
6. **Year Summary**: 2-3 sentence summary of the year

Return valid JSON:
{
  "year": "YYYY",
  "projects": [{"name": "", "role": "", "key_outcomes": []}],
  "top_accomplishments": [{"description": "", "category": "", "impact": ""}],
  "feedback": {"positive": [], "constructive": []},
  "key_relationships": [{"person": "", "relationship": ""}],
  "themes": [],
  "year_summary": ""
}"""


def consolidate(
    client: anthropic.Anthropic,
    quarters: list[str],
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    batch_delay: float = 60.0,
) -> dict:
    """Run Pass 2: hierarchical consolidation (by year, then final)."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    result_path = SUMMARY_DIR / "consolidated.json"

    existing = safe_json_load(str(result_path))
    if existing and not existing.get("parse_error"):
        log.info("Consolidated analysis already exists, skipping.")
        return existing

    # Collect quarter analyses grouped by year
    by_year: dict[str, list] = {}
    for q in sorted(quarters):
        analysis = safe_json_load(str(ANALYSIS_DIR / f"{q}_analysis.json"))
        if analysis:
            year = q.split("_")[0]
            by_year.setdefault(year, []).append(analysis)

    if not by_year:
        log.error("No quarter analyses found for consolidation")
        return {}

    # Phase 1: Consolidate each year
    year_summaries = []
    for year in sorted(by_year.keys()):
        year_path = SUMMARY_DIR / f"{year}_summary.json"
        cached = safe_json_load(str(year_path))
        if cached and not cached.get("parse_error"):
            log.info(f"Year {year} summary cached, skipping.")
            year_summaries.append(cached)
            continue

        quarters_data = by_year[year]
        compact = [_compact_quarter_summary(q) for q in quarters_data]
        summaries_text = json.dumps(compact, indent=1, default=str)

        if dry_run:
            year_summaries.append({"year": year, "dry_run": True})
            continue

        user_msg = f"Year: {year}\n\nQuarterly analyses:\n{summaries_text}\n\nConsolidate into a yearly summary."
        log.info(f"Consolidating year {year}: {len(quarters_data)} quarters, "
                 f"~{estimate_tokens(user_msg)}tok input")

        raw = _call_claude(client, YEAR_CONSOLIDATION_SYSTEM, user_msg, model=model, max_tokens=8192)
        result = _parse_json_response(raw)
        result["year"] = year
        safe_json_dump(result, str(year_path))
        year_summaries.append(result)

        time.sleep(batch_delay)

    if dry_run:
        return {"years": len(year_summaries), "dry_run": True}

    # Phase 2: Final consolidation of yearly summaries
    summaries_text = ""
    for ys in year_summaries:
        summaries_text += f"\n\n=== {ys.get('year', 'unknown')} ===\n"
        summaries_text += json.dumps(ys, indent=1, default=str)

    user_msg = PASS2_USER_TEMPLATE.format(quarterly_summaries=summaries_text)
    log.info(f"Final consolidation: {len(year_summaries)} years, "
             f"~{estimate_tokens(user_msg)}tok input")

    raw = _call_claude(
        client, PASS2_SYSTEM, user_msg,
        model=model, max_tokens=16384,
    )
    result = _parse_json_response(raw)
    safe_json_dump(result, str(result_path))

    log.info("Consolidation complete.")
    return result


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    model: str = DEFAULT_MODEL,
    quarter_filter: Optional[str] = None,
    batch_delay: float = 1.0,
    dry_run: bool = False,
    parsed_dir: Optional[str] = None,
) -> dict:
    """Run the full two-pass analysis pipeline."""
    client = anthropic.Anthropic()

    quarters = get_parsed_quarters(parsed_dir)
    if quarter_filter:
        quarters = [q for q in quarters if q == quarter_filter]

    if not quarters:
        log.error("No parsed quarters found. Run parsing first.")
        return {}

    log.info(f"Analysis: {len(quarters)} quarters to process")

    # Pass 1: per-quarter analysis
    pass1_results = {}
    for quarter in tqdm(quarters, desc="Analyzing quarters"):
        result = analyze_quarter(
            client, quarter,
            model=model,
            batch_delay=batch_delay,
            dry_run=dry_run,
            parsed_dir=parsed_dir,
        )
        pass1_results[quarter] = result

    if dry_run:
        total_input = sum(r.get("estimated_input_tokens", 0) for r in pass1_results.values())
        total_batches = sum(r.get("batch_count", 0) for r in pass1_results.values())
        pass2_est = consolidate(client, quarters, model=model, dry_run=True)
        total_input += pass2_est.get("estimated_input_tokens", 0)
        total_cost = round(total_input * 3 / 1_000_000, 2)
        # Add estimated output cost
        est_output_tokens = total_batches * BATCH_OUTPUT_TOKENS + 16384
        total_cost += round(est_output_tokens * 15 / 1_000_000, 2)

        return {
            "dry_run": True,
            "quarters": len(quarters),
            "total_batches": total_batches,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": est_output_tokens,
            "estimated_cost_usd": total_cost,
            "per_quarter": pass1_results,
        }

    # Pass 2: consolidation
    consolidated = consolidate(client, quarters, model=model)

    return {
        "quarters_analyzed": len(pass1_results),
        "consolidated": bool(consolidated),
    }
