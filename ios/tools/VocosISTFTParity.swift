import Foundation

@main
struct VocosISTFTParity {
    static func readFloats(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw NSError(domain: "VocosISTFTParity", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid float fixture size: \(path)"
            ])
        }
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            fatalError("Usage: VocosISTFTParity real_imag.f32 waveform.f32")
        }
        let spectrum = try readFloats(CommandLine.arguments[1])
        let reference = try readFloats(CommandLine.arguments[2])
        let actual = try VocosISTFT.decode(
            realImag: spectrum,
            frameCount: 16,
            fftSize: 640,
            hopLength: 160,
            paddingMode: .center
        )
        guard actual.count == reference.count else {
            fatalError("Sample count mismatch: Swift=\(actual.count), PyTorch=\(reference.count)")
        }

        var maxAbsoluteError: Float = 0
        var squaredError: Double = 0
        for index in actual.indices {
            let error = abs(actual[index] - reference[index])
            maxAbsoluteError = max(maxAbsoluteError, error)
            squaredError += Double(error * error)
        }
        let rmsError = sqrt(squaredError / Double(max(1, actual.count)))
        print("Vocos Accelerate iSTFT parity: samples=\(actual.count), max_abs=\(maxAbsoluteError), rms=\(rmsError)")
        guard maxAbsoluteError < 1e-3 && rmsError < 1e-4 else {
            fatalError("Accelerate iSTFT does not match the PyTorch reference")
        }
        print("VOCOS_ISTFT_PARITY_PASSED")
    }
}
