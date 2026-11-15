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
    enum Padding: Equatable {
        case center
        case same
    }

    static func decode(
        realImag: [Float],
        frameCount: Int,
        fftSize: Int = 640,
        hopLength: Int = 160,
        paddingMode: Padding = .center
    ) throws -> [Float] {
        let frequencyBins = fftSize / 2 + 1
        guard frameCount > 0,
              fftSize > 0,
              hopLength > 0,
              hopLength <= fftSize,
              realImag.count == frequencyBins * frameCount * 2 else {
            throw VocosDecoderError.invalidSpectrogramShape
        }

        // Vocos uses a 640-point transform (5 * 2^7), not a power-of-two FFT.
        // vDSP_DFT supports this length and keeps the inverse transform exact.
        guard let setup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.INVERSE
        ) else {
            throw VocosDecoderError.unsupportedFFTSize
        }
        defer { vDSP_DFT_DestroySetup(setup) }

        // torch.hann_window(win_length) uses the periodic Hann definition.
        let window = (0..<fftSize).map { index -> Float in
            0.5 - 0.5 * cos(2.0 * Float.pi * Float(index) / Float(fftSize))
        }
        let trimPadding = paddingMode == .center
            ? fftSize / 2
            : (fftSize - hopLength) / 2
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

            var timeReal = [Float](repeating: 0, count: fftSize)
            var timeImag = [Float](repeating: 0, count: fftSize)
            real.withUnsafeBufferPointer { realBuffer in
                imag.withUnsafeBufferPointer { imagBuffer in
                    timeReal.withUnsafeMutableBufferPointer { realOutput in
                        timeImag.withUnsafeMutableBufferPointer { imagOutput in
                            guard let inputReal = realBuffer.baseAddress,
                                  let inputImag = imagBuffer.baseAddress,
                                  let outputReal = realOutput.baseAddress,
                                  let outputImag = imagOutput.baseAddress else { return }
                            vDSP_DFT_Execute(
                                setup,
                                inputReal,
                                inputImag,
                                outputReal,
                                outputImag
                            )
                            // vDSP's inverse DFT is unnormalized; PyTorch irfft divides by N.
                            var scale = 1.0 / Float(fftSize)
                            vDSP_vsmul(
                                outputReal, 1,
                                &scale,
                                outputReal, 1,
                                vDSP_Length(fftSize)
                            )
                        }
                    }
                }
            }

            let destinationStart = frame * hopLength
            for sample in 0..<fftSize {
                let weight = window[sample]
                output[destinationStart + sample] += timeReal[sample] * weight
                envelope[destinationStart + sample] += weight * weight
            }
        }

        // MeanVC2's Vocos checkpoint uses same padding.
        let lower = trimPadding
        let upper = outputSize - trimPadding
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
    private let paddingMode: VocosISTFT.Padding

    init(
        modelURL: URL,
        fftSize: Int = 640,
        hopLength: Int = 160,
        paddingMode: VocosISTFT.Padding = .center
    ) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let loadedModel = try MLModel(contentsOf: modelURL, configuration: configuration)
        guard let input = loadedModel.modelDescription.inputDescriptionsByName.keys.first,
              let output = loadedModel.modelDescription.outputDescriptionsByName.keys.first else {
            throw VocosDecoderError.invalidModelOutput
        }
        self.model = loadedModel
        self.inputName = input
        self.outputName = output
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.paddingMode = paddingMode
    }

    /// mel is the MeanVC2 decoder output in [-1, 1], channel-major [80, frames].
    /// The reference runtime maps it to [0, 1] before calling Vocos.
    func decode(mel: [Float], frameCount: Int) throws -> [Float] {
        guard frameCount > 0, mel.count == 80 * frameCount else {
            throw VocosDecoderError.invalidSpectrogramShape
        }

        let input = try MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: 80), NSNumber(value: frameCount)],
            dataType: .float32
        )
        for index in mel.indices {
            input[index] = NSNumber(value: (mel[index] + 1.0) * 0.5)
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
            hopLength: hopLength,
            paddingMode: paddingMode
        )
    }
}
