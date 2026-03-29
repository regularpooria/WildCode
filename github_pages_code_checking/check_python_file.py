#!/usr/bin/env python3

"""Validate one Python file against many YAML rules.

This script reads Semgrep-style YAML files, prepares rule matchers using
`extract_patterns.py`, filters to Python/generic rules, and scans a target file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from extract_patterns import (
    find_yaml_files,
    prepare_rule_matchers,
    read_rules_from_file,
)

try:
    import regex as timeout_regex
except Exception:  # noqa: BLE001
    timeout_regex = None


def build_regex(raw_pattern: str) -> re.Pattern[str] | None:
    """Build a Python regex while supporting simple leading inline flags.

    Example supported prefixes: (?i), (?m), (?im)
    """
    if not raw_pattern:
        return None

    pattern = raw_pattern
    flags = 0

    while pattern.startswith("(?") and len(pattern) > 3:
        close = pattern.find(")")
        if close <= 2:
            break

        token = pattern[2:close]
        if not token.isalpha():
            break

        for ch in token:
            if ch == "i":
                flags |= re.IGNORECASE
            elif ch == "m":
                flags |= re.MULTILINE
            elif ch == "s":
                flags |= re.DOTALL

        pattern = pattern[close + 1 :]

    try:
        return re.compile(pattern, flags)
    except re.error:
        return None


def load_python_rules(rules_dir: Path) -> list[dict[str, Any]]:
    """Load, normalize, and filter rules relevant to Python scans."""
    output: list[dict[str, Any]] = []

    for yaml_file in find_yaml_files(rules_dir):
        rel_source = str(yaml_file.relative_to(rules_dir))
        for rule in read_rules_from_file(yaml_file):
            prepared = prepare_rule_matchers(rule)
            prepared["source_file"] = prepared.get("source_file", rel_source)

            langs = [str(v).strip().lower() for v in prepared.get("languages", [])]
            if "python" in langs or "generic" in langs:
                output.append(prepared)

    return output


def scan_python_file(
    py_file: Path,
    rules: list[dict[str, Any]],
    search_timeout_ms: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Return matching rule hits for the provided Python file.

    Returns:
        findings: Matched rules.
        compile_failures: Invalid regex pattern count.
        timeout_count: Regex searches that timed out.
    """
    content = py_file.read_text(encoding="utf-8", errors="replace")
    findings: list[dict[str, str]] = []
    compile_failures = 0
    timeout_count = 0

    for rule in rules:
        regex_list = [x for x in rule.get("match_regexes", []) if isinstance(x, str)]
        if not regex_list:
            continue

        for raw_regex in regex_list:
            compiled = build_regex(raw_regex)
            if compiled is None:
                compile_failures += 1
                continue

            found = None
            if timeout_regex is not None:
                try:
                    # regex module uses seconds for timeout.
                    found = timeout_regex.search(
                        compiled.pattern,
                        content,
                        flags=compiled.flags,
                        timeout=search_timeout_ms / 1000.0,
                    )
                except TimeoutError:
                    timeout_count += 1
                    continue
                except Exception:
                    # Fall back to stdlib search when regex module rejects flags.
                    found = compiled.search(content)
            else:
                found = compiled.search(content)

            if found is None:
                continue

            findings.append(
                {
                    "id": str(rule.get("id", "unknown-id")),
                    "severity": str(rule.get("severity", "INFO")),
                    "message": str(rule.get("message", "")),
                    "source_file": str(rule.get("source_file", "unknown")),
                    "match": found.group(0).strip().replace("\n", " ")[:160],
                }
            )
            break

    return findings, compile_failures, timeout_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan one Python file against a directory of YAML rules."
    )
    parser.add_argument("--rules-dir", required=True, help="Directory with YAML rules")
    parser.add_argument(
        "--python-file", required=True, help="Target Python file to scan"
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=50,
        help="Max findings to print (default: 50)",
    )
    parser.add_argument(
        "--search-timeout-ms",
        type=int,
        default=30,
        help="Per-regex search timeout in milliseconds (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rules_dir = Path(args.rules_dir).expanduser().resolve()
    py_file = Path(args.python_file).expanduser().resolve()

    if not rules_dir.is_dir():
        raise SystemExit(f"Rules directory does not exist: {rules_dir}")
    if not py_file.is_file():
        raise SystemExit(f"Python file does not exist: {py_file}")

    rules = load_python_rules(rules_dir)
    findings, compile_failures, timeout_count = scan_python_file(
        py_file,
        rules,
        search_timeout_ms=args.search_timeout_ms,
    )

    print(f"Loaded Python/generic rules: {len(rules)}")
    print(f"Findings: {len(findings)}")
    print(f"Regex compile failures: {compile_failures}")
    print(f"Regex timeouts: {timeout_count}")

    for item in findings[: args.max_findings]:
        print("-")
        print(f"id: {item['id']}")
        print(f"severity: {item['severity']}")
        print(f"source: {item['source_file']}")
        if item["message"]:
            print(f"message: {item['message']}")
        if item["match"]:
            print(f"match: {item['match']}")


if __name__ == "__main__":
    main()
