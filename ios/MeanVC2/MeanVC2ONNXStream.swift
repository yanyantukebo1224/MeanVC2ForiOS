import Foundation

enum MeanVC2ONNXError: Error {
    case modelMissing(String)
    case invalidTensor(String)
    case targetVoiceNotSet
}

/// Native Swift host loop for the published 120ms_40ms MeanVC2 ONNX graphs.
/// Input is mono Float32 audio at 16 kHz. Each call may contain one 160 ms
/// chunk; output is the audio produced so far and can initially be empty.
final class MeanVC2ONNXStream: @unchecked Sendable {
    private static let sampleRate = 16_000
    private static let inputChunkSamples = 2_560
    private static let fbankFrameSamples = 400
    private static let fbankStride = 160
    private static let bnWindow = 19
    private static let bnStride = 16
    private static let requiredCache = 8
    private static let chunkFrames = 12
    private static let lookaheadFrames = 4
    private static let vocoderOverlapFrames = 2
    private static let waveformOverlapSamples = 160
    private static let layerCount = 4

    private let sessions: [String: ORTSession]
    private let odeSteps: Int

    private var speaker: ORTValue?
    private var memoryKeys: ORTValue?
    private var memoryValues: ORTValue?

    private var sampleCache: [Float] = []
    private var frameCache: [Float] = []
    private var encoderOutputCache: [Float] = []
    private var asrOffset = 8
    private var attentionCache: ORTValue
    private var convolutionCache: ORTValue
    private var keyValueCaches: [ORTValue] = []
    private var vcOffset = 0
    private var noiseCache: [Float]?
    private var vocoderCache: [Float]?
    private var lastWaveform: [Float]?
    private var bnBuffer: [Float] = []
    // Production inference gets fresh Gaussian noise; CI can inject a fixed noiseProvider.
    private var randomState: UInt64 = UInt64.random(in: UInt64.min...UInt64.max)
    var noiseProvider: ((Int) -> [Float])?

    /// The default solver schedule matches runtime/run_rt.py's 120ms profile (N = 2).
    init(modelDirectory: URL, threads: Int = 2, odeSteps: Int = 2) throws {
        self.odeSteps = max(1, odeSteps)
        var loaded: [String: ORTSession] = [:]
        for name in ["fbank", "asr", "gtm", "dit", "vocos"] {
            let fileName: String
            switch name {
            case "asr": fileName = "asr.int8.onnx"
            case "dit": fileName = "dit.int8.onnx"
            case "vocos": fileName = "vocos.int8.onnx"
            default: fileName = "\(name).onnx"
            }
            let url = modelDirectory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MeanVC2ONNXError.modelMissing(fileName)
            }
            loaded[name] = try ONNXRuntimeEnvironment.makeSession(modelURL: url, threads: threads)
        }
        self.sessions = loaded
        self.attentionCache = try ORTTensor.float(
            [Float](repeating: 0, count: 6 * 4 * Self.requiredCache * 128),
            shape: [6, 4, Self.requiredCache, 128]
        )
        self.convolutionCache = try ORTTensor.float(
            [Float](repeating: 0, count: 6 * 1 * 256 * 8),
            shape: [6, 1, 256, 8]
        )
        try resetKeyValueCaches()
    }

    /// Sets the 256-dimensional target-speaker embedding and clears stream state.
    func setTargetEmbedding(_ embedding: [Float]) throws {
        if let validationError = MeanVC2SpeakerEmbeddingValidator.validationError(for: embedding) {
            throw MeanVC2ONNXError.invalidTensor(validationError)
        }
        speaker = try ORTTensor.float(embedding, shape: [1, 256])
        let result = try run("gtm", inputs: ["spks": speaker!], outputs: ["k_mem", "v_mem"])
        guard let keys = result["k_mem"], let values = result["v_mem"] else {
            throw MeanVC2ONNXError.invalidTensor("GTM output is incomplete")
        }
        memoryKeys = keys
        memoryValues = values
        try reset()
    }

    /// Clears streaming caches while retaining the selected target voice.
    func reset() throws {
        sampleCache.removeAll(keepingCapacity: true)
        frameCache.removeAll(keepingCapacity: true)
        encoderOutputCache.removeAll(keepingCapacity: true)
        asrOffset = 8
        attentionCache = try ORTTensor.float(
            [Float](repeating: 0, count: 6 * 4 * Self.requiredCache * 128),
            shape: [6, 4, Self.requiredCache, 128]
        )
        convolutionCache = try ORTTensor.float(
            [Float](repeating: 0, count: 6 * 1 * 256 * 8),
            shape: [6, 1, 256, 8]
        )
        try resetKeyValueCaches()
        vcOffset = 0
        noiseCache = nil
        vocoderCache = nil
        lastWaveform = nil
        bnBuffer.removeAll(keepingCapacity: true)
    }

    /// Processes the next mono 16 kHz input samples. For a live stream, call
    /// serially with 2,560 samples per call and queue returned audio for playback.
    func process(_ samples: [Float]) throws -> [Float] {
        guard speaker != nil, memoryKeys != nil, memoryValues != nil else {
            throw MeanVC2ONNXError.targetVoiceNotSet
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw MeanVC2ONNXError.invalidTensor("input contains non-finite samples")
        }
        guard let newBN = try encodeChunk(samples) else { return [] }
        bnBuffer.append(contentsOf: newBN)

        let requiredFrames = Self.chunkFrames + Self.lookaheadFrames
        var output: [Float] = []
        for _ in 0..<4 {
            guard bnBuffer.count >= requiredFrames * 256 else { break }
            let condition = Array(bnBuffer.prefix(requiredFrames * 256))
            bnBuffer.removeFirst(Self.chunkFrames * 256)
            let mel = try convertStep(condition: condition)
            output.append(contentsOf: try decodeMel(mel, frames: Self.chunkFrames))
        }
        return output
    }

    // MARK: - ASR and fbank

    private func encodeChunk(_ samples: [Float]) throws -> [Float]? {
        var padded = sampleCache
        padded.append(contentsOf: samples)
        guard padded.count >= Self.fbankFrameSamples else {
            sampleCache = padded
            return nil
        }

        let frameCount = 1 + (padded.count - Self.fbankFrameSamples) / Self.fbankStride
        var frames = [Float](repeating: 0, count: frameCount * Self.fbankFrameSamples)
        for frame in 0..<frameCount {
            let start = frame * Self.fbankStride
            for index in 0..<Self.fbankFrameSamples {
                frames[frame * Self.fbankFrameSamples + index] =
                    padded[start + index] * 32_768.0
            }
        }

        let fbankInput = try ORTTensor.float(
            frames, shape: [frameCount, Self.fbankFrameSamples]
        )
        let fbankOutput = try run(
            "fbank", inputs: ["frames": fbankInput], outputs: ["fbank"]
        )
        guard let fbankValue = fbankOutput["fbank"] else {
            throw MeanVC2ONNXError.invalidTensor("fbank output is missing")
        }
        var features = try ORTTensor.readFloat(fbankValue)
        sampleCache = Array(padded[(Self.fbankStride * frameCount)...])
        if !frameCache.isEmpty {
            features = frameCache + features
        }

        if asrOffset >= 4_000 {
            asrOffset = max(Self.requiredCache, asrOffset - 4_000)
        }

        var bottleneckChunks: [[Float]] = []
        var cursor = 0
        while cursor + Self.bnWindow <= features.count / 80 {
            let window = Array(
                features[(cursor * 80)..<((cursor + Self.bnWindow) * 80)]
            )
            let fbank = try ORTTensor.float(window, shape: [1, Self.bnWindow, 80])
            let offset = try ORTTensor.int64Scalar(Int64(asrOffset))
            let cacheSize = try ORTTensor.int64Scalar(Int64(Self.requiredCache))
            let result = try run(
                "asr",
                inputs: [
                    "xs": fbank,
                    "offset": offset,
                    "required_cache_size": cacheSize,
                    "att_cache": attentionCache,
                    "cnn_cache": convolutionCache
                ],
                outputs: ["bn", "new_att_cache", "new_cnn_cache"]
            )
            guard let bn = result["bn"],
                  let nextAttention = result["new_att_cache"],
                  let nextConvolution = result["new_cnn_cache"] else {
                throw MeanVC2ONNXError.invalidTensor("ASR output is incomplete")
            }
            bottleneckChunks.append(try ORTTensor.readFloat(bn))
            attentionCache = nextAttention
            convolutionCache = nextConvolution
            asrOffset += 4
            cursor += Self.bnStride
        }

        frameCache = Array(features[(cursor * 80)...])
        guard !bottleneckChunks.isEmpty else { return nil }

        var bottleneck = bottleneckChunks.flatMap { $0 }
        if !encoderOutputCache.isEmpty {
            bottleneck = encoderOutputCache + bottleneck
        }
        let rowCount = bottleneck.count / 256
        encoderOutputCache = Array(bottleneck[((rowCount - 1) * 256)..<(rowCount * 256)])
        guard rowCount >= 2 else { return nil }
        return Self.interpolateAlignCorners(
            bottleneck, rows: rowCount, outputRows: 17, columns: 256
        ).dropFirst(256).map { $0 }
    }

    // MARK: - DiT flow-matching step

    private func convertStep(condition: [Float]) throws -> [Float] {
        let frameCount = Self.chunkFrames + Self.lookaheadFrames
        var x = gaussianNoise(count: frameCount * 80)
        if let noiseCache {
            x.replaceSubrange(0..<(Self.lookaheadFrames * 80), with: noiseCache)
        }
        noiseCache = Array(x.suffix(Self.lookaheadFrames * 80))

        if vcOffset >= 4_000 {
            vcOffset = 0
            try resetKeyValueCaches()
        }

        let conditionValue = try ORTTensor.float(condition, shape: [1, frameCount, 256])
        let offsetValue = try ORTTensor.int64Scalar(Int64(vcOffset))
        var inputs: [String: ORTValue] = [
            "cond": conditionValue,
            "spks": speaker!,
            "k_mem": memoryKeys!,
            "v_mem": memoryValues!,
            "offset": offsetValue
        ]
        for layer in 0..<Self.layerCount {
            inputs["k_cache\(layer)"] = keyValueCaches[layer * 2]
            inputs["v_cache\(layer)"] = keyValueCaches[layer * 2 + 1]
        }

        let dt = Float(1.0 / Double(odeSteps))
        for step in 0..<odeSteps {
            let tValue = Float(1.0 - Double(step) / Double(odeSteps))
            let rValue = max(0, tValue - dt)
            inputs["x"] = try ORTTensor.float(x, shape: [1, frameCount, 80])
            inputs["t"] = try ORTTensor.float([tValue], shape: [1])
            inputs["r"] = try ORTTensor.float([rValue], shape: [1])

            let names = Set(
                ["u"] + (0..<Self.layerCount).flatMap {
                    ["new_k_cache\($0)", "new_v_cache\($0)"]
                }
            )
            let result = try run("dit", inputs: inputs, outputs: names)
            guard let velocity = result["u"] else {
                throw MeanVC2ONNXError.invalidTensor("DiT output is missing")
            }
            let velocityValues = try ORTTensor.readFloat(velocity)
            guard velocityValues.count == x.count else {
                throw MeanVC2ONNXError.invalidTensor("DiT output shape is invalid")
            }
            for index in x.indices {
                x[index] -= dt * velocityValues[index]
            }
            if step == odeSteps - 1 {
                var nextCaches: [ORTValue] = []
                for layer in 0..<Self.layerCount {
                    guard let key = result["new_k_cache\(layer)"],
                          let value = result["new_v_cache\(layer)"] else {
                        throw MeanVC2ONNXError.invalidTensor("DiT cache output is missing")
                    }
                    nextCaches.append(key)
                    nextCaches.append(value)
                }
                keyValueCaches = nextCaches
            }
        }

        vcOffset += Self.chunkFrames
        var mel = [Float](repeating: 0, count: 80 * Self.chunkFrames)
        for time in 0..<Self.chunkFrames {
            for channel in 0..<80 {
                mel[channel * Self.chunkFrames + time] = x[time * 80 + channel]
            }
        }
        return mel
    }

    // MARK: - Vocos

    private func decodeMel(_ mel: [Float], frames: Int) throws -> [Float] {
        var combined = mel
        var combinedFrames = frames
        if let vocoderCache {
            combined = [Float](repeating: 0, count: 80 * (Self.vocoderOverlapFrames + frames))
            combinedFrames += Self.vocoderOverlapFrames
            for channel in 0..<80 {
                for frame in 0..<Self.vocoderOverlapFrames {
                    combined[channel * combinedFrames + frame] =
                        vocoderCache[channel * Self.vocoderOverlapFrames + frame]
                }
                for frame in 0..<frames {
                    combined[channel * combinedFrames + Self.vocoderOverlapFrames + frame] =
                        mel[channel * frames + frame]
                }
            }
        }

        var nextCache = [Float](repeating: 0, count: 80 * Self.vocoderOverlapFrames)
        for channel in 0..<80 {
            for frame in 0..<Self.vocoderOverlapFrames {
                nextCache[channel * Self.vocoderOverlapFrames + frame] =
                    combined[channel * combinedFrames + combinedFrames - Self.vocoderOverlapFrames + frame]
            }
        }
        vocoderCache = nextCache

        let normalized = combined.map { ($0 + 1.0) * 0.5 }
        let input = try ORTTensor.float(normalized, shape: [1, 80, combinedFrames])
        let result = try run("vocos", inputs: ["mel": input], outputs: ["wav"])
        guard let waveformValue = result["wav"] else {
            throw MeanVC2ONNXError.invalidTensor("Vocos output is missing")
        }
        let waveform = try ORTTensor.readFloat(waveformValue)
        let overlap = Self.waveformOverlapSamples
        guard waveform.count > overlap else {
            throw MeanVC2ONNXError.invalidTensor("Vocos waveform is too short")
        }

        let output: [Float]
        if let lastWaveform {
            var joined = [Float](repeating: 0, count: overlap)
            for index in 0..<overlap {
                let amount = Float(index) / Float(overlap - 1)
                joined[index] = lastWaveform[index] * (1 - amount) + waveform[index] * amount
            }
            let middleEnd = waveform.count - overlap
            if middleEnd > overlap {
                joined.append(contentsOf: waveform[overlap..<middleEnd])
            }
            output = joined
        } else {
            output = Array(waveform.dropLast(overlap))
        }
        lastWaveform = Array(waveform.suffix(overlap))
        return output
    }

    // MARK: - Helpers

    private func run(
        _ name: String,
        inputs: [String: ORTValue],
        outputs: Set<String>
    ) throws -> [String: ORTValue] {
        guard let session = sessions[name] else {
            throw MeanVC2ONNXError.modelMissing(name)
        }
        let result = try session.run(
            withInputs: inputs,
            outputNames: outputs,
            runOptions: nil
        )
        return result
    }

    private func resetKeyValueCaches() throws {
        keyValueCaches = try (0..<(2 * Self.layerCount)).map { _ in
            try ORTTensor.float([], shape: [1, 2, 0, 64])
        }
    }

    private func gaussianNoise(count: Int) -> [Float] {
        if let noiseProvider {
            let supplied = noiseProvider(count / 80)
            if supplied.count == count { return supplied }
        }
        var values = [Float](repeating: 0, count: count)
        var index = 0
        while index < count {
            let u1 = max(Double.leastNonzeroMagnitude, 1.0 - nextRandom())
            let u2 = nextRandom()
            let radius = sqrt(-2.0 * log(u1))
            let angle = 2.0 * Double.pi * u2
            values[index] = Float(radius * cos(angle))
            if index + 1 < count {
                values[index + 1] = Float(radius * sin(angle))
            }
            index += 2
        }
        return values
    }

    private func nextRandom() -> Double {
        randomState = 6_364_136_223_846_793_005 &* randomState &+ 1_442_695_040_888_963_407
        return Double(randomState >> 11) / Double(1 << 53)
    }

    private static func interpolateAlignCorners(
        _ values: [Float], rows: Int, outputRows: Int, columns: Int
    ) -> [Float] {
        guard rows > 1, outputRows > 1 else {
            return Array(repeating: values, count: outputRows).flatMap { $0 }
        }
        var output = [Float](repeating: 0, count: outputRows * columns)
        let scale = Float(rows - 1) / Float(outputRows - 1)
        for row in 0..<outputRows {
            let position = Float(row) * scale
            let lower = min(Int(floor(position)), rows - 2)
            let weight = position - Float(lower)
            for column in 0..<columns {
                let a = values[lower * columns + column]
                let b = values[(lower + 1) * columns + column]
                output[row * columns + column] = a * (1 - weight) + b * weight
            }
        }
        return output
    }
}

enum ORTTensor {
    static func float(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData()
        values.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress, bytes.count > 0 {
                data.append(base, length: bytes.count)
            }
        }
        return try ORTValue(
            tensorData: data,
            elementType: .float,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    static func int64Scalar(_ value: Int64) throws -> ORTValue {
        var mutableValue = value
        let data = NSMutableData(bytes: &mutableValue, length: MemoryLayout<Int64>.size)
        return try ORTValue(tensorData: data, elementType: .int64, shape: [])
    }

    static func readFloat(_ value: ORTValue) throws -> [Float] {
        let data = try value.tensorData()
        let count = data.length / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        let pointer = data.bytes.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
