import Foundation

/// Shared deterministic, non-speech fixture for Simulator graph execution.
/// This is test input only; it is never used for normal user conversion.
enum MeanVC2CISmokeFixture {
    static let chunkCount = 8
    static let chunkSamples = 2_560
    private static let speakerNorm = 1_158.3781765900117

    static func waveformChunk(index: Int) -> [Float] {
        precondition((0..<chunkCount).contains(index))
        let start = index * chunkSamples
        return (0..<chunkSamples).map { offset in
            let sample = start + offset
            let low = (sample * 37 + 11) % 257 - 128
            let high = (sample * 13 + 7) % 127 - 63
            return Float(low) / 2_048.0 + Float(high) / 4_096.0
        }
    }

    /// A deterministic unit-length vector that exercises GTM conditioning.
    /// It is not a speaker identity and must never be used as a target voice.
    static func speakerEmbedding() -> [Float] {
        (0..<256).map { index in
            let value = (index * 29 + 11) % 251 - 125
            return Float(Double(value) / speakerNorm)
        }
    }

    /// Deterministic non-zero noise in row-major [frames, 80] layout.
    static func noise(frameCount: Int) -> [Float] {
        precondition(frameCount >= 0)
        return (0..<(frameCount * 80)).map { index in
            Float((index * 17 + 3) % 101 - 50) / 512.0
        }
    }
}
