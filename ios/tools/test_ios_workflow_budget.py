#!/usr/bin/env python3
"""Guard the macOS integration workflow against silent 25-minute cancellations."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ios-build.yml"


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")

    job = re.search(
        r"(?ms)^jobs:\s*\n\s+build:\s*\n.*?^\s+runs-on:\s*macos-15\s*\n\s+timeout-minutes:\s*(\d+)",
        text,
    )
    assert job, "could not find macOS build job timeout"
    assert int(job.group(1)) >= 60, (
        f"macOS integration job budget is only {job.group(1)} minutes"
    )

    lines = text.splitlines()
    steps: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 6 and stripped.startswith("- name: "):
            name = stripped[len("- name: "):]
            end = index + 1
            while end < len(lines):
                candidate = lines[end]
                candidate_stripped = candidate.lstrip()
                candidate_indent = len(candidate) - len(candidate_stripped)
                if candidate_indent == 6 and candidate_stripped.startswith("- name: "):
                    break
                end += 1
            steps[name] = "\n".join(lines[index + 1:end])
            index = end
        else:
            index += 1

    required_budgets = {
        "Verify ONNX Runtime Objective-C source archive": 10,
        "Install CocoaPods dependencies": 15,
        "Build iOS Simulator app": 20,
        "Download and verify ONNX smoke-test models": 12,
        "Run ONNX streaming inference smoke test in iPhone Simulator": 20,
        "Install matching ONNX Runtime for reference parity": 10,
        "Check Swift streaming output against Python reference": 15,
    }
    for name, minimum in required_budgets.items():
        body = steps.get(name)
        assert body is not None, f"missing workflow step: {name}"
        match = re.search(r"(?m)^        timeout-minutes:\s*(\d+)\s*$", body)
        assert match, f"{name} has no step timeout"
        assert int(match.group(1)) >= minimum, (
            f"{name} timeout {match.group(1)} is below {minimum}"
        )

    model_downloads = [
        line for line in text.splitlines()
        if "$base/" in line and "?download=true" in line
    ]
    assert len(model_downloads) == 5, (
        f"expected 5 pinned model download commands, found {len(model_downloads)}"
    )
    for line in model_downloads:
        for flag in ("--connect-timeout 15", "--max-time 600", "--retry-all-errors"):
            assert flag in line, f"model download lacks {flag}: {line.strip()}"

    print(
        "iOS workflow budget OK: "
        f"job={job.group(1)}m, {len(required_budgets)} bounded long-running steps, "
        "5 bounded model downloads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
