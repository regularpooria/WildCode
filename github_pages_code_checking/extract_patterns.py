#!/usr/bin/env python3

"""Aggregate Semgrep-style YAML rule files into one JSON file.

Example:
	python github_pages_code_checking/extract_patterns.py \
		--input-dir opengrep-rules \
		--output-json combined_rules.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


def iter_nested_nodes(root: Any) -> list[Any]:
    """Return all nested list/dict nodes under root (including root)."""
    stack = [root]
    nodes: list[Any] = []

    while stack:
        node = stack.pop()
        nodes.append(node)

        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    return nodes


def collect_regex_strings(rule: dict[str, Any]) -> list[str]:
    """Collect direct regex fields from a Semgrep-like rule."""
    regexes: list[str] = []

    pattern_regex = rule.get("pattern-regex")
    if isinstance(pattern_regex, str) and pattern_regex:
        regexes.append(pattern_regex)

    fix_regex = rule.get("fix-regex")
    if isinstance(fix_regex, dict):
        raw = fix_regex.get("regex")
        if isinstance(raw, str) and raw:
            regexes.append(raw)

    for node in iter_nested_nodes(rule):
        if isinstance(node, dict):
            raw = node.get("pattern-regex")
            if isinstance(raw, str) and raw:
                regexes.append(raw)

    # Preserve order while deduplicating.
    return list(dict.fromkeys(regexes))


def collect_pattern_templates(rule: dict[str, Any]) -> list[str]:
    """Collect all string-valued `pattern` templates from a rule."""
    templates: list[str] = []

    for node in iter_nested_nodes(rule):
        if not isinstance(node, dict):
            continue
        raw = node.get("pattern")
        if isinstance(raw, str) and raw:
            templates.append(raw)

    return list(dict.fromkeys(templates))


def template_to_regex(template: str) -> str | None:
    """Convert a Semgrep-ish template into a broad regex approximation.

    This is intentionally permissive and optimized for quick browser scanning,
    not full AST-level Semgrep semantics.
    """
    normalized = template.strip()
    if not normalized:
        return None

    pattern = normalized

    # Ignore templates that are mostly metavariables/ellipsis with no stable anchors.
    literal_view = re.sub(r"\$[A-Z_][A-Z0-9_]*", " ", normalized)
    literal_view = literal_view.replace("...", " ")
    literal_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}", literal_view)
    long_tokens = [tok for tok in literal_tokens if len(tok) >= 4]
    if not long_tokens:
        return None

    # Replace Semgrep metavariables and ellipsis with placeholders first.
    pattern = re.sub(r"\$[A-Z_][A-Z0-9_]*", "__SEMGREP_VAR__", pattern)
    pattern = pattern.replace("...", "__SEMGREP_ELLIPSIS__")

    pattern = re.escape(pattern)
    # Keep metavariable matching bounded to avoid matching the entire file.
    pattern = pattern.replace("__SEMGREP_VAR__", r"[A-Za-z_][A-Za-z0-9_\.]{0,120}")
    # Keep ellipsis bounded for performance and precision.
    pattern = pattern.replace("__SEMGREP_ELLIPSIS__", r"[\s\S]{0,240}?")
    # Whitespace tolerant, but require at least some separation where space existed.
    pattern = pattern.replace(r"\ ", r"\s+")

    return pattern


def prepare_rule_matchers(rule: dict[str, Any]) -> dict[str, Any]:
    """Build normalized matcher metadata for a single rule."""
    raw_regexes = collect_regex_strings(rule)
    template_regexes = [
        x for x in (template_to_regex(t) for t in collect_pattern_templates(rule)) if x
    ]

    # Deduplicate while preserving order.
    merged = list(dict.fromkeys([*raw_regexes, *template_regexes]))
    rule["match_regexes"] = merged
    rule["matcher_stats"] = {
        "direct_regexes": len(raw_regexes),
        "template_regexes": len(template_regexes),
        "total_match_regexes": len(merged),
    }
    return rule


def find_yaml_files(input_dir: Path) -> list[Path]:
    """Return all .yml/.yaml files under input_dir (recursive)."""
    files = [*input_dir.rglob("*.yml"), *input_dir.rglob("*.yaml")]
    return sorted(files)


def read_rules_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Parse all YAML documents in file_path and collect rules entries."""
    with file_path.open("r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    rules: list[dict[str, Any]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        doc_rules = doc.get("rules")
        if not isinstance(doc_rules, list):
            continue

        for item in doc_rules:
            if isinstance(item, dict):
                rules.append(item)

    return rules


def aggregate_rules(
    input_dir: Path,
    include_source: bool = False,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Aggregate rules from all YAML files in input_dir.

    Returns:
            rules: Merged list of rule objects.
            errors: List of parsing errors with file path and reason.
    """
    all_rules: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    yaml_files = find_yaml_files(input_dir)
    for file_path in yaml_files:
        try:
            rules = read_rules_from_file(file_path)
            if include_source:
                rel_path = str(file_path.relative_to(input_dir))
                for rule in rules:
                    rule["source_file"] = rel_path
            all_rules.extend(rules)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - continue collecting others unless strict
            if strict:
                raise
            errors.append({"file": str(file_path), "error": str(exc)})

    return all_rules, errors


def build_rule_matchers(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach precomputed regex matcher lists for browser-side scanning."""
    prepared: list[dict[str, Any]] = []
    for rule in rules:
        prepared.append(prepare_rule_matchers(rule))
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse YAML rule files from a directory and save as one JSON file."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing YAML files.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Add source_file to each rule.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first YAML parsing error.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )

    rules, errors = aggregate_rules(
        input_dir=input_dir,
        include_source=args.include_source,
        strict=args.strict,
    )
    rules = build_rule_matchers(rules)

    total_match_regexes = sum(
        int(rule.get("matcher_stats", {}).get("total_match_regexes", 0))
        for rule in rules
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "input_directory": str(input_dir),
        "total_rules": len(rules),
        "total_match_regexes": total_match_regexes,
        "total_errors": len(errors),
        "rules": rules,
        "errors": errors,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=args.indent, ensure_ascii=False)
        f.write("\n")

    print(
        f"Wrote {len(rules)} rules from '{input_dir}' to '{output_json}' "
        f"({len(errors)} parse errors)."
    )


if __name__ == "__main__":
    main()
