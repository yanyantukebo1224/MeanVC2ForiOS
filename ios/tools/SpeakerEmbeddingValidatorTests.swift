import Foundation

@main
struct SpeakerEmbeddingValidatorTests {
    static func main() {
        var assertions = 0

        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            assertions += 1
            guard condition() else {
                fatalError("Speaker embedding validator test failed: \(message)")
            }
        }

        let valid = (0..<256).map { Float(($0 % 7) - 3) / 8.0 }
        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(for: valid) == nil,
            "finite non-zero 256-D vector should be accepted"
        )

        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(
                for: [Float](repeating: 1, count: 255)
            ) != nil,
            "short vector should be rejected"
        )
        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(
                for: [Float](repeating: 1, count: 257)
            ) != nil,
            "long vector should be rejected"
        )

        var withNaN = valid
        withNaN[17] = .nan
        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(for: withNaN) != nil,
            "NaN should be rejected"
        )

        var withInfinity = valid
        withInfinity[23] = .infinity
        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(for: withInfinity) != nil,
            "infinity should be rejected"
        )

        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(
                for: [Float](repeating: 0, count: 256)
            ) != nil,
            "zero vector should be rejected"
        )
        check(
            MeanVC2SpeakerEmbeddingValidator.validationError(
                for: [Float](repeating: 1e-9, count: 256)
            ) != nil,
            "negligible-energy vector should be rejected"
        )

        print("Speaker embedding validator tests passed (\(assertions) assertions)")
    }
}
