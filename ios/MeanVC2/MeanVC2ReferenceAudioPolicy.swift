import Foundation

/// Bounds reference audio to the range exercised by the published speaker
/// encoder checks and prevents a long recording from being fully decoded into
/// memory before validation.
enum MeanVC2ReferenceAudioPolicy {
    static let sampleRate = 16_000
    static let minimumDurationSeconds = 3.0
    static let maximumDurationSeconds = 20.0

    static func validationError(durationSeconds: Double) -> String? {
        guard durationSeconds.isFinite,
              durationSeconds >= minimumDurationSeconds,
              durationSeconds <= maximumDurationSeconds else {
            return "Reference audio must be between 3 and 20 seconds"
        }
        return nil
    }

    static func validationError(sampleCount: Int) -> String? {
        guard sampleCount >= minimumDurationSeconds * Double(sampleRate),
              sampleCount <= maximumDurationSeconds * Double(sampleRate) else {
            return "Reference audio must be between 3 and 20 seconds at 16 kHz"
        }
        return nil
    }
}
