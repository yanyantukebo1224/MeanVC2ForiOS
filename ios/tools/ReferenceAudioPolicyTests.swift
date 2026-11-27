import Foundation

@main
struct ReferenceAudioPolicyTests {
    static func main() {
        var assertions = 0

        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            assertions += 1
            guard condition() else {
                fatalError("Reference audio policy test failed: " + message)
            }
        }

        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: 3.0) == nil,
            "minimum supported duration should be accepted"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: 20.0) == nil,
            "maximum supported duration should be accepted"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: 2.999) != nil,
            "short duration should be rejected"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: 20.001) != nil,
            "long duration should be rejected"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: .nan) != nil,
            "NaN duration should be rejected"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(durationSeconds: .infinity) != nil,
            "infinite duration should be rejected"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(sampleCount: 48_000) == nil,
            "three seconds at 16 kHz should be accepted"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(sampleCount: 320_000) == nil,
            "twenty seconds at 16 kHz should be accepted"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(sampleCount: 47_999) != nil,
            "less than three seconds should be rejected"
        )
        check(
            MeanVC2ReferenceAudioPolicy.validationError(sampleCount: 320_001) != nil,
            "more than twenty seconds should be rejected"
        )

        print("Reference audio policy tests passed (" + String(assertions) + " assertions)")
    }
}
