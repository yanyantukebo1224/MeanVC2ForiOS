import AVFoundation
import Foundation

/// Captures microphone audio, resamples it to 16 kHz, runs the streaming
/// converter off the audio callback, then resamples and queues the result.
final class MeanVC2LiveAudioController {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let processingQueue = DispatchQueue(label: "jp.yanyantukebo.meanvc2.audio-processing", qos: .userInitiated)
    private let mono16k = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    )!

    private var inputConverter: AVAudioConverter?
    private var outputConverter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?
    private var stream: MeanVC2ONNXStream?
    private var pendingInput: [Float] = []
    private var playerAttached = false
    private(set) var isRunning = false

    var onError: ((String) -> Void)?

    func start(stream: MeanVC2ONNXStream) throws {
        guard !isRunning else { return }
        self.stream = stream

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .voiceChat,
            options: [.defaultToSpeaker, .allowBluetooth]
        )
        try session.setPreferredSampleRate(48_000)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        let hardwareOutputFormat = engine.outputNode.inputFormat(forBus: 0)
        guard let newInputConverter = AVAudioConverter(from: inputFormat, to: mono16k),
              let newOutputConverter = AVAudioConverter(from: mono16k, to: hardwareOutputFormat) else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }
        inputConverter = newInputConverter
        outputConverter = newOutputConverter
        outputFormat = hardwareOutputFormat

        if !playerAttached {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: hardwareOutputFormat)
            playerAttached = true
        }

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: inputFormat
        ) { [weak self] buffer, _ in
            self?.consumeMicrophoneBuffer(buffer)
        }

        engine.prepare()
        do {
            try engine.start()
            player.play()
            pendingInput.removeAll(keepingCapacity: true)
            isRunning = true
        } catch {
            inputNode.removeTap(onBus: 0)
            throw error
        }
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        player.stop()
        engine.stop()
        isRunning = false
        pendingInput.removeAll(keepingCapacity: true)
        inputConverter = nil
        outputConverter = nil
        stream = nil
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }

    private func consumeMicrophoneBuffer(_ input: AVAudioPCMBuffer) {
        guard let converter = inputConverter else { return }
        let ratio = mono16k.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * ratio)) + 512
        guard let converted = AVAudioPCMBuffer(
            pcmFormat: mono16k,
            frameCapacity: max(capacity, 512)
        ) else { return }

        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: converted, error: &conversionError) {
            _, inputStatus in
            if suppliedInput {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error,
              converted.frameLength > 0,
              let samples = converted.floatChannelData?.pointee else {
            if let conversionError {
                reportError(conversionError.localizedDescription)
            }
            return
        }

        let copy = Array(UnsafeBufferPointer(
            start: samples,
            count: Int(converted.frameLength)
        ))
        processingQueue.async { [weak self] in
            self?.processInput(copy)
        }
    }

    private func processInput(_ samples: [Float]) {
        guard let stream else { return }
        pendingInput.append(contentsOf: samples)
        while pendingInput.count >= 2_560 {
            let chunk = Array(pendingInput.prefix(2_560))
            pendingInput.removeFirst(2_560)
            do {
                let converted = try stream.process(chunk)
                if !converted.isEmpty {
                    try enqueueOutput(converted)
                }
            } catch {
                reportError(error.localizedDescription)
                return
            }
        }
    }

    private func enqueueOutput(_ samples: [Float]) throws {
        guard let converter = outputConverter,
              let outputFormat else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }
        guard let input = AVAudioPCMBuffer(
            pcmFormat: mono16k,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress,
               let destination = input.floatChannelData?.pointee {
                destination.update(from: base, count: samples.count)
            }
        }

        let ratio = outputFormat.sampleRate / mono16k.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(samples.count) * ratio)) + 1_024
        guard let output = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: max(capacity, 1_024)
        ) else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }

        var suppliedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) {
            _, inputStatus in
            if suppliedInput {
                inputStatus.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            inputStatus.pointee = .haveData
            return input
        }
        if let conversionError { throw conversionError }
        guard status != .error, output.frameLength > 0 else {
            throw MeanVC2AudioFileError.conversionFailed("Audio playback resampling failed")
        }
        player.scheduleBuffer(output, completionHandler: nil)
    }

    private func reportError(_ message: String) {
        onError?(message)
    }
}
