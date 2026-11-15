import Accelerate
import CoreML
import Foundation

enum VocosDecoderError: Error {
    case invalidSpectrogramShape
    case invalidModelOutput
    case unsupportedFFTSize
}

/// Reconstructs the waveform from Vocos' real/imaginary one-sided spectrum.
/// The learned Vocos backbone and spectral projection are exported to Core ML;
/// this overlap-add stage is native Accelerate code.
enum VocosISTFT {
    static func decode(
        realImag: [Float],
        frameCount: Int,
        fftSize: Int = 1024,
        hopLength: Int = 160
    ) throws -> [Float] {
        let frequencyBins = fftSize / 2 + 1
        guard frameCount > 0,
              fftSize > 0,
              fftSize.nonzeroBitCount == 1,
              hopLength > 0,
              hopLength <= fftSize,
              realImag.count == frequencyBins * frameCount * 2 else {
            throw VocosDecoderError.invalidSpectrogramShape
        }

        let log2FFT = vDSP_Length(fftSize.trailingZeroBitCount)
        guard let setup = vDSP_create_fftsetup(log2FFT, FFTRadix(kFFTRadix2)) else {
            throw VocosDecoderError.unsupportedFFTSize
        }
        defer { vDSP_destroy_fftsetup(setup) }

        // torch.hann_window(win_length) uses the periodic Hann definition.
        let window = (0..<fftSize).map { index -> Float in
            0.5 - 0.5 * cos(2.0 * Float.pi * Float(index) / Float(fftSize))
        }
        let padding = (fftSize - hopLength) / 2
        let outputSize = (frameCount - 1) * hopLength + fftSize
        var output = [Float](repeating: 0, count: outputSize)
        var envelope = [Float](repeating: 0, count: outputSize)

        for frame in 0..<frameCount {
            var real = [Float](repeating: 0, count: fftSize)
            var imag = [Float](repeating: 0, count: fftSize)

            // Core ML output layout: [1, frequency, time, real/imag].
            for bin in 0..<frequencyBins {
                let source = (bin * frameCount + frame) * 2
                real[bin] = realImag[source]
                imag[bin] = realImag[source + 1]

                // Complete the Hermitian spectrum required by a complex IFFT.
                if bin > 0 && bin < fftSize / 2 {
                    real[fftSize - bin] = real[bin]
                    imag[fftSize - bin] = -imag[bin]
                }
            }

            real.withUnsafeMutableBufferPointer { realBuffer in
                imag.withUnsafeMutableBufferPointer { imagBuffer in
                    guard let realBase = realBuffer.baseAddress,
                          let imagBase = imagBuffer.baseAddress else { return }
                    var split = DSPSplitComplex(realp: realBase, imagp: imagBase)
                    vDSP_fft_zip(
                        setup,
                        &split,
                        1,
                        log2FFT,
                        FFTDirection(FFT_INVERSE)
                    )
                    var scale = 1.0 / Float(fftSize)
                    vDSP_vsmul(
                        realBase, 1,
                        &scale,
                        realBase, 1,
                        vDSP_Length(fftSize)
                    )
                }
            }

            let destinationStart = frame * hopLength
            for sample in 0..<fftSize {
                let weight = window[sample]
                output[destinationStart + sample] += real[sample] * weight
                envelope[destinationStart + sample] += weight * weight
            }
        }

        // MeanVC2's Vocos checkpoint uses same padding.
        let lower = padding
        let upper = outputSize - padding
        guard upper > lower else { return [] }

        var waveform = [Float](repeating: 0, count: upper - lower)
        for index in lower..<upper {
            let norm = envelope[index]
            waveform[index - lower] = norm > 1e-11 ? output[index] / norm : 0
        }
        return waveform
    }
}

/// Loads the exported Core ML spectral model and performs native iSTFT synthesis.
final class VocosCoreMLDecoder {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let fftSize: Int
    private let hopLength: Int

    init(
        modelURL: URL,
        fftSize: Int = 1024,
        hopLength: Int = 160
    ) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        guard let input = model.modelDescription.inputDescriptionsByName.keys.first,
              let output = model.modelDescription.outputDescriptionsByName.keys.first else {
            throw VocosDecoderError.invalidModelOutput
        }
        self.inputName = input
        self.outputName = output
        self.fftSize = fftSize
        self.hopLength = hopLength
    }

    /// mel is channel-major [80, frames], normalized exactly as MeanVC2 runtime expects.
    func decode(mel: [Float], frameCount: Int) throws -> [Float] {
        guard frameCount > 0, mel.count == 80 * frameCount else {
            throw VocosDecoderError.invalidSpectrogramShape
        }

        let input = try MLMultiArray(
            shape: [1, 80, frameCount] as [NSNumber],
            dataType: .float32
        )
        for index in mel.indices {
            input[index] = NSNumber(value: mel[index])
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: input)
        ])
        let result = try model.prediction(from: provider)
        guard let spectrum = result.featureValue(for: outputName)?.multiArrayValue,
              spectrum.shape.count == 4,
              spectrum.shape[0].intValue == 1,
              spectrum.shape[3].intValue == 2 else {
            throw VocosDecoderError.invalidModelOutput
        }

        let bins = spectrum.shape[1].intValue
        let outputFrames = spectrum.shape[2].intValue
        guard bins == fftSize / 2 + 1 else {
            throw VocosDecoderError.invalidModelOutput
        }

        var realImag = [Float](repeating: 0, count: bins * outputFrames * 2)
        for index in realImag.indices {
            realImag[index] = spectrum[index].floatValue
        }
        return try VocosISTFT.decode(
            realImag: realImag,
            frameCount: outputFrames,
            fftSize: fftSize,
            hopLength: hopLength
        )
    }
}
