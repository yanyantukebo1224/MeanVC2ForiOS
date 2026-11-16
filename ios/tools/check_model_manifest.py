#!/usr/bin/env python3
"""Keep the Swift model store, CI downloads, and third-party notices in sync."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ASSET_RE = re.compile(
    r'Asset\s*\(\s*name:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,'
    r'\s*sha256:\s*"([0-9a-f]{64})"\s*\)',
    re.DOTALL,
)


def fail(message: str) -> None:
    raise SystemExit(f"model manifest check failed: {message}")


def parse_assets(block: str) -> list[tuple[str, int, str]]:
    return [
        (name, int(size), digest)
        for name, size, digest in ASSET_RE.findall(block)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--verify-files",
        type=Path,
        help="also verify downloaded model sizes and SHA-256 digests in this directory",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    store_path = root / "ios/MeanVC2/MeanVC2ModelStore.swift"
    workflow_path = root / ".github/workflows/ios-build.yml"
    notices_path = root / "THIRD_PARTY_NOTICES.md"
    store = store_path.read_text(encoding="utf-8")
    workflow = workflow_path.read_text(encoding="utf-8")
    notices = notices_path.read_text(encoding="utf-8")

    revision_match = re.search(
        r'static let revision = "([0-9a-f]{40})"', store
    )
    workflow_revision_match = re.search(
        r'base="https://huggingface\.co/kiarina/MeanVC2-ONNX/resolve/([0-9a-f]{40})"',
        workflow,
    )
    if not revision_match or not workflow_revision_match:
        fail("could not find both pinned Hugging Face revisions")
    revision = revision_match.group(1)
    if revision != workflow_revision_match.group(1):
        fail("Swift and CI use different Hugging Face revisions")

    conversion_block_match = re.search(
        r"static let conversionAssets:\s*\[Asset\]\s*=\s*\[(.*?)\n\s*\]",
        store,
        re.DOTALL,
    )
    if not conversion_block_match:
        fail("could not find conversionAssets in MeanVC2ModelStore.swift")
    assets = parse_assets(conversion_block_match.group(1))
    if len(assets) != 5:
        fail(f"expected 5 conversion assets, found {len(assets)}")
    names = [name for name, _, _ in assets]
    if len(set(names)) != len(names):
        fail("duplicate conversion asset names")

    speaker_start = store.find("static let speakerAsset")
    if speaker_start < 0:
        fail("optional speakerAsset declaration is missing")
    speaker_assets = parse_assets(store[speaker_start:])
    if len(speaker_assets) != 1 or speaker_assets[0][0] != "speaker/speaker.int8.onnx":
        fail("optional WavLM + ECAPA speaker asset declaration is invalid")
    if speaker_assets[0][1] <= 0 or "CC BY-SA 3.0" not in notices:
        fail("speaker asset size or separate CC BY-SA 3.0 attribution is missing")

    downloads = re.findall(r"\$base/([^?\s\"]+)\?download=true", workflow)
    checksums = {
        name: digest
        for digest, name in re.findall(
            r'echo "([0-9a-f]{64})\s+([^"]+)"\s*\|\s*shasum -a 256 -c -',
            workflow,
        )
    }
    if downloads != names:
        fail(f"CI download list differs from Swift asset list: {downloads!r}")
    for name, size, digest in assets:
        if size <= 0:
            fail(f"{name} has an invalid byte count")
        if checksums.get(name) != digest:
            fail(f"CI checksum for {name} differs from Swift")

    if args.verify_files:
        model_dir = args.verify_files.resolve()
        for name, expected_size, expected_digest in assets:
            path = model_dir / name
            if not path.is_file():
                fail(f"downloaded model is missing: {path}")
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                fail(f"{name}: expected {expected_size} bytes, got {actual_size}")
            actual_digest = sha256(path)
            if actual_digest != expected_digest:
                fail(f"{name}: SHA-256 mismatch ({actual_digest})")

    total_bytes = sum(size for _, size, _ in assets)
    print(
        "Model manifest OK: "
        f"{len(assets)} conversion graphs, {total_bytes} bytes, "
        f"revision {revision}; optional speaker encoder separately licensed."
    )
    if args.verify_files:
        print(f"Downloaded model files verified in {args.verify_files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
