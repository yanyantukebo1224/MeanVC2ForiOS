#!/usr/bin/env python3
"""Ensure both speaker-vector ingress paths use the shared validator."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"speaker embedding validation check failed: {message}")


def main() -> int:
    app = ROOT / "ios/MeanVC2"
    validator = (app / "MeanVC2SpeakerEmbeddingValidator.swift").read_text(encoding="utf-8")
    stream = (app / "MeanVC2ONNXStream.swift").read_text(encoding="utf-8")
    encoder = (app / "MeanVC2SpeakerEncoder.swift").read_text(encoding="utf-8")
    project = (app / "MeanVC2.xcodeproj/project.pbxproj").read_text(encoding="utf-8")
    tests = (ROOT / "ios/tools/SpeakerEmbeddingValidatorTests.swift").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ios-build.yml").read_text(encoding="utf-8")

    require("static let dimensions = 256" in validator, "validator dimension must match GTM")
    require("value.isFinite" in validator, "validator must reject NaN and infinity")
    require("minimumSquaredNorm" in validator, "validator must reject zero-energy vectors")
    require("MeanVC2SpeakerEmbeddingValidator.validationError(for: embedding)" in stream,
            "GTM ingress must validate restored and newly registered embeddings")
    require("MeanVC2SpeakerEmbeddingValidator.validationError(for: vector)" in encoder,
            "speaker encoder output must be validated before it is returned")
    require(stream.index("validationError(for: embedding)") < stream.index("ORTTensor.float(embedding"),
            "GTM validation must happen before tensor creation")
    require(encoder.index("validationError(for: vector)") < encoder.index("return vector"),
            "speaker validation must happen before returning the embedding")
    require("MeanVC2SpeakerEmbeddingValidator.swift in Sources" in project,
            "validator must be compiled into the app target")
    require("MeanVC2SpeakerEmbeddingValidator.swift" in project,
            "validator file reference is missing")
    for marker in (".nan", ".infinity", "count: 255", "count: 257", "repeating: 0", "1e-9"):
        require(marker in tests, f"missing focused regression case: {marker}")
    require("SpeakerEmbeddingValidatorTests.swift" in workflow,
            "macOS CI must compile and run the Swift validator tests")

    print("Speaker embedding validation integration OK: encoder output and GTM input guarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
