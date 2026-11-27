import Foundation

/// Validates the target-speaker vector before it is passed to GTM.
/// The vector is deliberately not normalized here: the Python reference
/// consumes the speaker encoder output as-is.
enum MeanVC2SpeakerEmbeddingValidator {
    static let dimensions = 256
    private static let minimumSquaredNorm = 1e-12

    /// Returns a user-readable validation failure, or nil for a usable vector.
    static func validationError(for embedding: [Float]) -> String? {
        guard embedding.count == dimensions else {
            return "speaker embedding must contain exactly 256 values"
        }

        var squaredNorm = 0.0
        for value in embedding {
            guard value.isFinite else {
                return "speaker embedding contains a non-finite value"
            }
            let sample = Double(value)
            squaredNorm += sample * sample
        }

        guard squaredNorm.isFinite, squaredNorm > minimumSquaredNorm else {
            return "speaker embedding has zero or negligible energy"
        }
        return nil
    }
}
