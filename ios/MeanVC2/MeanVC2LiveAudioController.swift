import AVFoundation
import Foundation

/// Captures microphone audio, resamples it to 16 kHz, runs the streaming
/// converter off the audio callback, then resamples and queues the result.
final class MeanVC2LiveAudioController {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let processingQueue = DispatchQueue(
        label: "jp.yanyantukebo.meanvc2.audio-processing",
        qos: .userInitiated
    )
    private let inputWorkSlots = DispatchSemaphore(value: 24)
    private let maximumQueuedAudioSeconds = 0.6
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
    private var activeStream: MeanVC2ONNXStream?
    private var pendingInput: [Float] = []
    private var outputFrameBudget = MeanVC2AudioFrameBudget(maximumFrames: 28_800)
    private var playbackGeneration: UInt64 = 0
    private var playerAttached = false
    private var tapInstalled = false
    private(set) var isRunning = false

    private var interruptionObserver: NSObjectProtocol?
    private var routeChangeObserver: NSObjectProtocol?

    var onError: ((String) -> Void)?
    var onRunningChanged: ((Bool, String) -> Void)?

    init() {
        let session = AVAudioSession.sharedInstance()
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: session,
            queue: .main
        ) { [weak self] notification in
            self?.handleInterruption(notification)
        }
        routeChangeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: session,
            queue: .main
        ) { [weak self] notification in
            self?.handleRouteChange(notification)
        }
    }

    deinit {
        if let interruptionObserver {
            NotificationCenter.default.removeObserver(interruptionObserver)
        }
        if let routeChangeObserver {
            NotificationCenter.default.removeObserver(routeChangeObserver)
        }
    }

    func start(stream: MeanVC2ONNXStream) throws {
        guard !isRunning else { return }

        // A new capture session must not inherit ASR, DiT, noise, or vocoder
        // cache state from a previous run or an audio-route restart.
        try stream.reset()
        self.stream = stream
        activeStream = stream

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

        processingQueue.sync {
            pendingInput.removeAll(keepingCapacity: true)
            outputFrameBudget = MeanVC2AudioFrameBudget(
                maximumFrames: max(
                    1,
                    Int(hardwareOutputFormat.sampleRate * maximumQueuedAudioSeconds)
                )
            )
        }

        if playerAttached {
            engine.disconnectNodeOutput(player)
        } else {
            engine.attach(player)
            playerAttached = true
        }
        engine.connect(player, to: engine.mainMixerNode, format: hardwareOutputFormat)

        inputNode.installTap(
            onBus: 0,
            bufferSize: 1_024,
            format: inputFormat
        ) { [weak self] buffer, _ in
            self?.consumeMicrophoneBuffer(buffer)
        }
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
            player.play()
            isRunning = true
            onRunningChanged?(true, "")
        } catch {
            engine.stop()
            if tapInstalled {
                inputNode.removeTap(onBus: 0)
                tapInstalled = false
            }
            player.stop()
            inputConverter = nil
            outputConverter = nil
            outputFormat = nil
            isRunning = false
            throw error
        }
    }

    func stop() {
        stopEngine(clearStream: true, deactivateSession: true)
        onRunningChanged?(false, "")
    }

    private func stopEngine(clearStream: Bool, deactivateSession: Bool) {
        if engine.isRunning {
            engine.stop()
        }
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        player.stop()
        isRunning = false

        processingQueue.sync {
            playbackGeneration &+= 1
            outputFrameBudget.reset()
            pendingInput.removeAll(keepingCapacity: true)
            if clearStream {
                stream = nil
            }
        }

        inputConverter = nil
        outputConverter = nil
        outputFormat = nil
        if clearStream {
            activeStream = nil
        }
        if deactivateSession {
            try? AVAudioSession.sharedInstance().setActive(
                false,
                options: .notifyOthersOnDeactivation
            )
        }
    }

    private func handleInterruption(_ notification: Notification) {
        guard let rawType = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: rawType) else {
            return
        }

        switch type {
        case .began:
            guard isRunning else { return }
            stopEngine(clearStream: false, deactivateSession: false)
            onRunningChanged?(false, "音声セッションが中断されました。再開を待っています")
        case .ended:
            guard let activeStream else { return }
            let rawOptions = notification.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
            let options = AVAudioSession.InterruptionOptions(rawValue: rawOptions)
            guard options.contains(.shouldResume) else {
                onRunningChanged?(false, "音声セッションが終了しました。必要なら再度開始してください")
                return
            }
            do {
                try start(stream: activeStream)
            } catch {
                onRunningChanged?(false, "音声セッションを再開できませんでした")
                reportError(error.localizedDescription)
            }
        @unknown default:
            break
        }
    }

    private func handleRouteChange(_ notification: Notification) {
        guard isRunning, let activeStream,
              let rawReason = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: rawReason) else {
            return
        }

        let shouldRestart =
            reason == .oldDeviceUnavailable ||
            reason == .newDeviceAvailable ||
            reason == .categoryChange ||
            reason == .override
        guard shouldRestart else { return }

        stopEngine(clearStream: false, deactivateSession: false)
        do {
            try start(stream: activeStream)
        } catch {
            onRunningChanged?(false, "音声出力先の変更後に再開できませんでした")
            reportError(error.localizedDescription)
        }
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
        let slots = inputWorkSlots
        guard slots.wait(timeout: .now()) == .success else {
            // Keep callback-to-inference memory bounded when the model cannot
            // keep up with realtime input. The next accepted buffer continues
            // through the same streaming state.
            return
        }
        processingQueue.async { [weak self, slots] in
            defer { slots.signal() }
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

        let frameCount = Int(output.frameLength)
        guard outputFrameBudget.reserve(frameCount) else {
            // Drop newest output instead of allowing the playback queue and
            // end-to-end latency to grow without bound.
            return
        }
        let generation = playbackGeneration
        player.scheduleBuffer(output, completionHandler: { [weak self] in
            guard let self else { return }
            self.processingQueue.async {
                guard self.playbackGeneration == generation else { return }
                self.outputFrameBudget.release(frameCount)
            }
        })
    }

    private func reportError(_ message: String) {
        onError?(message)
    }
}
