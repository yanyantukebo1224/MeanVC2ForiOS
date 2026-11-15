import Foundation
import OnnxRuntimeBindings

/// One-time reference-voice embedding extraction. The speaker graph is a
/// separately licensed optional asset; see THIRD_PARTY_NOTICES.md.
final class MeanVC2SpeakerEncoder {
    private let session: ORTSession

    init(modelURL: URL, threads: Int = 2) throws {
        self.session = try ONNXRuntimeEnvironment.makeSession(
            modelURL: modelURL, threads: threads
        )
    }

    /// Accepts mono Float32 samples resampled to 16 kHz.
    func embedding(for waveform: [Float]) throws -> [Float] {
        guard waveform.count >= 3 * 16_000,
              waveform.allSatisfy(\.isFinite) else {
            throw MeanVC2ONNXError.invalidTensor(
                "Reference audio must contain at least three seconds of finite 16 kHz mono audio"
            )
        }
        let input = try ORTTensor.float(waveform, shape: [1, waveform.count])
        guard let result = try session.run(
            withInputs: ["wav": input],
            outputNames: ["embedding"],
            runOptions: nil
        ), let value = result["embedding"] else {
            throw MeanVC2ONNXError.invalidTensor("Speaker embedding output is missing")
        }
        let vector = try ORTTensor.readFloat(value)
        guard vector.count == 256 else {
            throw MeanVC2ONNXError.invalidTensor("Speaker embedding must contain 256 values")
        }
        return vector
    }
}
