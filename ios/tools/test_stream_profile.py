#!/usr/bin/env python3
"""Regression tests for parsing fbank dimensions from the real Kaldi call."""

from __future__ import annotations

import unittest

from check_stream_profile import fbank_sample_counts


class FbankProfileParserTests(unittest.TestCase):
    def test_uses_kaldi_arguments_instead_of_derived_values_in_comments(self) -> None:
        reference = """
        # The streaming frame grid uses frame_length=400 / frame_shift=160 samples.
        fbanks = kaldi.fbank(
            wav_t,
            frame_length=25,
            frame_shift=10,
            snip_edges=True,
            num_mel_bins=80,
        )
        """
        self.assertEqual(fbank_sample_counts(reference, 16_000), (400, 160))

    def test_handles_whitespace_and_multiline_arguments(self) -> None:
        reference = """
        fbanks = kaldi.fbank(
            wav_t,
            frame_length = 25,
            frame_shift = 10,
        )
        """
        self.assertEqual(fbank_sample_counts(reference, 48_000), (1_200, 480))

    def test_rejects_missing_kaldi_call(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Could not locate the Kaldi fbank call"):
            fbank_sample_counts("# frame_length=25 frame_shift=10", 16_000)


if __name__ == "__main__":
    unittest.main()
