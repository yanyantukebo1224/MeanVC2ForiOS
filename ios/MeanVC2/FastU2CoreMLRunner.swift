import CoreML
import Foundation

enum FastU2CoreMLError: Error {
    case invalidModel
    case invalidPositionalEncoding
    case invalidInput
    case invalidOutput
}

/// Stateful 40ms_40ms Fast-U2++ runner. The caller supplies Kaldi-compatible
/// 80-bin fbank windows; this class owns the streaming attention/CNN caches
/// and the absolute positional-encoding offset.
final class FastU2CoreMLRunner {
    private let model: MLModel
    private let positionalEncoding: [Float]
    private let positionalFrameCount: Int
    private let featureDimension = 256
    private let cacheLayers = 6
    private let cacheHeads = 4
    private let cacheFrames = 4
    private let headDimension = 128
    private let cnnChannels = 256
    private let cnnHistory = 8

    private var offset = 4
    private var attentionCache: [Float]
    private var convolutionCache: [Float]

    init(modelURL: URL, positionalEncodingURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)

        let data = try Data(contentsOf: positionalEncodingURL)
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw FastU2CoreMLError.invalidPositionalEncoding
        }
        let values = data.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        guard values.count >= 6 * featureDimension,
              values.count % featureDimension == 0 else {
            throw FastU2CoreMLError.invalidPositionalEncoding
        }
        self.positionalEncoding = values
        self.positionalFrameCount = values.count / featureDimension

        self.attentionCache = [Float](repeating: 0,
            count: cacheLayers * cacheHeads * cacheFrames * headDimension)
        self.convolutionCache = [Float](repeating: 0,
            count: cacheLayers * 1 * cnnChannels * cnnHistory)
    }

    /// Processes one 11-frame fbank window and returns the 2-frame, 256-dim
    /// bottleneck representation in time-major order.
    func encode(fbank: [Float]) throws -> [Float] {
        guard fbank.count == 11 * 80 else { throw FastU2CoreMLError.invalidInput }

        let positionStart = offset - cacheFrames
        let positionCount = cacheFrames + 2
        guard positionStart >= 0,
              positionStart + positionCount <= positionalFrameCount else {
            throw FastU2CoreMLError.invalidPositionalEncoding
        }

        let fbankArray = try makeArray(shape: [1, 11, 80], values: fbank)
        let positionValues = Array(
            positionalEncoding[(positionStart * featureDimension)..<((positionStart + positionCount) * featureDimension)]
        )
        let positionArray = try makeArray(shape: [1, positionCount, featureDimension], values: positionValues)
        let attentionArray = try makeArray(
            shape: [cacheLayers, cacheHeads, cacheFrames, headDimension],
            values: attentionCache
        )
        let convolutionArray = try makeArray(
            shape: [cacheLayers, 1, cnnChannels, cnnHistory],
            values: convolutionCache
        )

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "fbank": MLFeatureValue(multiArray: fbankArray),
            "pos_emb": MLFeatureValue(multiArray: positionArray),
            "att_cache": MLFeatureValue(multiArray: attentionArray),
            "cnn_cache": MLFeatureValue(multiArray: convolutionArray)
        ])
        let result = try model.prediction(from: provider)

        guard let bnf = result.featureValue(for: "bnf")?.multiArrayValue,
              let nextAttention = result.featureValue(for: "attention_cache_out")?.multiArrayValue,
              let nextConvolution = result.featureValue(for: "convolution_cache_out")?.multiArrayValue else {
            throw FastU2CoreMLError.invalidOutput
        }

        let bnfValues = floatValues(bnf)
        guard bnfValues.count == 2 * featureDimension,
              nextAttention.count == attentionCache.count,
              nextConvolution.count == convolutionCache.count else {
            throw FastU2CoreMLError.invalidOutput
        }

        attentionCache = floatValues(nextAttention)
        convolutionCache = floatValues(nextConvolution)
        offset += 2
        return bnfValues
    }

    func reset() {
        offset = 4
        attentionCache = [Float](repeating: 0, count: attentionCache.count)
        convolutionCache = [Float](repeating: 0, count: convolutionCache.count)
    }

    private func makeArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        guard array.count == values.count else { throw FastU2CoreMLError.invalidInput }
        for index in values.indices {
            array[index] = NSNumber(value: values[index])
        }
        return array
    }

    private func floatValues(_ array: MLMultiArray) -> [Float] {
        (0..<array.count).map { array[$0].floatValue }
    }
}
