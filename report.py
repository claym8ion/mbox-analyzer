"""Markdown report generator from consolidated analysis."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import log, safe_json_load

SUMMARY_DIR = Path(__file__).parent / "output" / "summaries"
ANALYSIS_DIR = Path(__file__).parent / "output" / "analysis"
REPORT_PATH = Path(__file__).parent / "output" / "report.md"


def generate_report(
    output_path: Optional[str] = None,
    parse_stats: Optional[dict] = None,
    analysis_stats: Optional[dict] = None,
) -> str:
    """Generate a Markdown report from consolidated analysis.

    Returns the path to the written report.
    """
    output_path = Path(output_path) if output_path else REPORT_PATH
    consolidated = safe_json_load(str(SUMMARY_DIR / "consolidated.json"))

    if not consolidated:
        log.error("No consolidated analysis found. Run analysis first.")
        return ""

    sections = []

    # Title
    sections.append("# Career Profile: Clay at LegitScript\n")
    sections.append(f"*Generated {datetime.now().strftime('%Y-%m-%d')} from email archive analysis*\n")

    # Executive Summary
    exec_summary = consolidated.get("executive_summary", "")
    if exec_summary:
        sections.append("## Executive Summary\n")
        sections.append(exec_summary + "\n")

    # Career Arc
    career_arc = consolidated.get("career_arc", {})
    if career_arc:
        sections.append("## Career Arc\n")
        trajectory = career_arc.get("overall_trajectory", "")
        if trajectory:
            sections.append(trajectory + "\n")
        phases = career_arc.get("phases", [])
        if phases:
            sections.append("### Phases\n")
            for phase in phases:
                period = phase.get("period", "")
                title = phase.get("title_or_role", "")
                focus = phase.get("focus", "")
                shifts = phase.get("key_shifts", "")
                sections.append(f"**{period}** — {title}\n")
                if focus:
                    sections.append(f"- **Focus:** {focus}")
                if shifts:
                    sections.append(f"- **Key shifts:** {shifts}")
                sections.append("")

    # Projects
    projects = consolidated.get("projects", [])
    if projects:
        sections.append("## Projects\n")
        for proj in projects:
            name = proj.get("name", "Unnamed")
            desc = proj.get("description", "")
            role = proj.get("role", "")
            timeline = proj.get("timeline", "")
            collabs = proj.get("collaborators", [])
            outcomes = proj.get("key_outcomes", [])
            status = proj.get("status", "")

            sections.append(f"### {name}\n")
            if timeline:
                sections.append(f"**Timeline:** {timeline}  ")
            if role:
                sections.append(f"**Role:** {role}  ")
            if status:
                sections.append(f"**Status:** {status}\n")
            if desc:
                sections.append(f"\n{desc}\n")
            if collabs:
                sections.append(f"**Collaborators:** {', '.join(collabs)}\n")
            if outcomes:
                sections.append("**Key Outcomes:**\n")
                for outcome in outcomes:
                    sections.append(f"- {outcome}")
                sections.append("")

    # Top Accomplishments
    accomplishments = consolidated.get("top_accomplishments", [])
    if accomplishments:
        sections.append("## Top Accomplishments\n")
        # Group by category
        by_category: dict[str, list] = {}
        for acc in accomplishments:
            cat = acc.get("category", "Other") or "Other"
            by_category.setdefault(cat, []).append(acc)

        for cat, items in sorted(by_category.items()):
            sections.append(f"### {cat}\n")
            for item in items:
                desc = item.get("description", "")
                quarter = item.get("quarter", "")
                impact = item.get("impact", "")
                line = f"- **{desc}**"
                if quarter:
                    line += f" ({quarter})"
                sections.append(line)
                if impact:
                    sections.append(f"  - *Impact:* {impact}")
            sections.append("")

    # Feedback & Recognition
    feedback = consolidated.get("feedback_summary", {})
    if feedback:
        sections.append("## Feedback & Recognition\n")

        strengths = feedback.get("strengths_identified", [])
        if strengths:
            sections.append("### Strengths Identified\n")
            for s in strengths:
                sections.append(f"- {s}")
            sections.append("")

        growth = feedback.get("growth_areas", [])
        if growth:
            sections.append("### Growth Areas\n")
            for g in growth:
                sections.append(f"- {g}")
            sections.append("")

        positive = feedback.get("notable_positive", [])
        if positive:
            sections.append("### Notable Positive Feedback\n")
            for fb in positive:
                who = fb.get("from", "")
                content = fb.get("content", "")
                context = fb.get("context", "")
                sections.append(f"- **{who}**: \"{content}\"")
                if context:
                    sections.append(f"  - *Context:* {context}")
            sections.append("")

        constructive = feedback.get("notable_constructive", [])
        if constructive:
            sections.append("### Notable Constructive Feedback\n")
            for fb in constructive:
                who = fb.get("from", "")
                content = fb.get("content", "")
                context = fb.get("context", "")
                sections.append(f"- **{who}**: \"{content}\"")
                if context:
                    sections.append(f"  - *Context:* {context}")
            sections.append("")

        given_themes = feedback.get("feedback_given_themes", [])
        if given_themes:
            sections.append("### Feedback Given to Others — Themes\n")
            for t in given_themes:
                sections.append(f"- {t}")
            sections.append("")

    # Key Relationships
    relationships = consolidated.get("key_relationships", [])
    if relationships:
        sections.append("## Key Relationships\n")
        sections.append("| Person | Relationship | Significance |")
        sections.append("|--------|-------------|--------------|")
        for rel in relationships:
            person = rel.get("person", "")
            relationship = rel.get("relationship", "")
            significance = rel.get("significance", "")
            sections.append(f"| {person} | {relationship} | {significance} |")
        sections.append("")

    # Themes
    themes = consolidated.get("themes", [])
    if themes:
        sections.append("## Major Themes\n")
        for theme in themes:
            sections.append(f"- {theme}")
        sections.append("")

    # Pipeline Statistics
    sections.append("## Appendix: Pipeline Statistics\n")
    if parse_stats:
        sections.append("### Parsing\n")
        sections.append(f"- **Emails scanned:** {parse_stats.get('total_scanned', 'N/A')}")
        sections.append(f"- **Emails kept:** {parse_stats.get('kept', 'N/A')}")
        sections.append(f"- **Excluded (spam):** {parse_stats.get('excluded_spam', 0)}")
        sections.append(f"- **Excluded (promotions):** {parse_stats.get('excluded_promotions', 0)}")
        sections.append(f"- **Excluded (calendar):** {parse_stats.get('excluded_calendar', 0)}")
        sections.append(f"- **No date:** {parse_stats.get('no_date', 0)}")
        sections.append(f"- **Quarters:** {parse_stats.get('quarters_written', 'N/A')}")
        qc = parse_stats.get("quarter_counts", {})
        if qc:
            sections.append("\n**Emails per quarter:**\n")
            sections.append("| Quarter | Count |")
            sections.append("|---------|-------|")
            for q, count in sorted(qc.items()):
                sections.append(f"| {q} | {count:,} |")
        sections.append("")

    if analysis_stats:
        sections.append("### Analysis\n")
        if analysis_stats.get("dry_run"):
            sections.append("*(Dry run — no API calls made)*\n")
        sections.append(f"- **Quarters analyzed:** {analysis_stats.get('quarters_analyzed', 'N/A')}")
        sections.append(f"- **Consolidation:** {'Complete' if analysis_stats.get('consolidated') else 'N/A'}")
        sections.append("")

    # Write report
    report_text = "\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info(f"Report written to {output_path} ({len(report_text):,} chars)")
    return str(output_path)
