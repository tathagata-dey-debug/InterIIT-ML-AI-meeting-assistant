"""
Dual JSON & Markdown export serialisers.
Serializes MeetingRecord into formatted JSON and executive Markdown reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

from .schemas import MeetingMinutes, MeetingRecord


def export_to_json(record: MeetingRecord) -> str:
    """
    Returns serialized JSON with 2-space indentation.

    Args:
        record: MeetingRecord instance.

    Returns:
        JSON string indented with 2 spaces.
    """
    return record.model_dump_json(indent=2)


def export_to_markdown(record: MeetingRecord) -> str:
    """
    Generates a formatted Markdown report featuring:
    # Meeting Record
    ## Executive Summary
    ## Discussion Points
    ## Key Decisions (if empty, print "No explicit decisions were finalized.")
    ## Action Items (Markdown table with columns: Task | Assignee | Deadline. Show 'Unspecified' clearly.)
    ---
    ## Refined Transcript
    ## Raw Transcript

    Args:
        record: MeetingRecord instance.

    Returns:
        Formatted Markdown string.
    """
    minutes = record.minutes

    md_lines = [
        "# Meeting Record",
        "",
        "## Executive Summary",
        minutes.concise_summary.strip(),
        "",
        "## Discussion Points",
    ]

    if minutes.discussion_points:
        for point in minutes.discussion_points:
            md_lines.append(f"- {point.strip()}")
        md_lines.append("")
    else:
        md_lines.append("*No discussion points recorded.*")
        md_lines.append("")

    md_lines.append("## Key Decisions")
    if minutes.key_decisions:
        for idx, dec in enumerate(minutes.key_decisions, start=1):
            quote = dec.supporting_quote.strip().replace('"', '\\"')
            md_lines.append(f"### {idx}. {dec.decision_text.strip()}")
            md_lines.append(f"- **Supporting Quote:** *\"{quote}\"*")
            md_lines.append("")
    else:
        md_lines.append("No explicit decisions were finalized.")
        md_lines.append("")

    md_lines.append("## Action Items")
    if minutes.action_items:
        md_lines.append("| Task | Assignee | Deadline |")
        md_lines.append("| :--- | :--- | :--- |")
        for item in minutes.action_items:
            task = item.task_description.strip().replace("|", "/")
            assignee = item.owner.strip() if item.owner else "Unspecified"
            deadline = item.deadline.strip() if item.deadline else "Unspecified"
            md_lines.append(f"| {task} | {assignee} | {deadline} |")
        md_lines.append("")
    else:
        md_lines.append("No actionable tasks were assigned.")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Refined Transcript",
        record.refined_transcript.strip(),
        "",
        "## Raw Transcript",
        record.raw_transcript.strip(),
    ])

    return "\n".join(md_lines)


class Exporter:
    """Encapsulates JSON and Markdown export serialization methods for MeetingRecord."""

    @staticmethod
    def export_to_json(record: MeetingRecord) -> str:
        """Serialize MeetingRecord to formatted JSON string."""
        return export_to_json(record)

    @staticmethod
    def export_to_markdown(record: MeetingRecord) -> str:
        """Serialize MeetingRecord into executive Markdown report."""
        return export_to_markdown(record)

    @classmethod
    def to_json(cls, data: Union[MeetingRecord, MeetingMinutes], indent: int = 2) -> str:
        """Backwards-compatible JSON serializer supporting both MeetingRecord and MeetingMinutes."""
        if isinstance(data, MeetingRecord):
            return export_to_json(data)
        return data.model_dump_json(indent=indent)

    @classmethod
    def to_markdown(cls, data: Union[MeetingRecord, MeetingMinutes]) -> str:
        """Backwards-compatible Markdown serializer supporting both MeetingRecord and MeetingMinutes."""
        if isinstance(data, MeetingRecord):
            return export_to_markdown(data)
        # Wrap standalone MeetingMinutes into dummy record if needed
        dummy = MeetingRecord(
            raw_transcript="",
            refined_transcript="",
            minutes=data,
        )
        return export_to_markdown(dummy)

    @classmethod
    def save_artifacts(
        cls,
        record: MeetingRecord,
        output_dir: Path | str,
        base_filename: str = "meeting_record",
    ) -> Tuple[Path, Path]:
        """
        Export both JSON and Markdown files to disk ensuring identical information.

        Args:
            record: Validated MeetingRecord instance.
            output_dir: Destination folder.
            base_filename: Base filename without extension.

        Returns:
            Tuple of (json_path, markdown_path).
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / f"{base_filename}.json"
        md_file = out_path / f"{base_filename}.md"

        json_content = export_to_json(record)
        md_content = export_to_markdown(record)

        json_file.write_text(json_content, encoding="utf-8")
        md_file.write_text(md_content, encoding="utf-8")

        return json_file, md_file
