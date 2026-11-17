import SwiftUI
import UniformTypeIdentifiers
import os

private enum MeanVC2SmokeDiagnostics {
    static func write(_ status: String) {
        #if DEBUG
        guard let documents = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first else { return }
        try? FileManager.default.createDirectory(
            at: documents, withIntermediateDirectories: true
        )
        let statusURL = documents.appendingPathComponent("onnx-pipeline-smoke-test.status")
        try? Data(status.utf8).write(to: statusURL, options: .atomic)
        NSLog("[MeanVC2Smoke] %@", status)
        #endif
    }
}

struct ContentView: View {
    @State private var referenceURL: URL?
    @State private var showingImporter = false
    @State private var showingModelInfo = false
    @State private var showingSpeakerConfirmation = false
    @State private var engineReady = false
    @State private var targetReady = false
    @State private var modelStatus = "モデルを準備しています"
    @State private var modelError: String?
    @State private var stream: MeanVC2ONNXStream?
    @State private var liveAudio = MeanVC2LiveAudioController()
    @State private var isLive = false

    private let accent = Color(red: 0.63, green: 0.48, blue: 1.0)
    private let modelLogger = Logger(subsystem: "jp.yanyantukebo.meanvc2", category: "Engine")

    var body: some View {
        ZStack {
            Color(red: 0.045, green: 0.047, blue: 0.075)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                header
                    .padding(.top, 12)

                Spacer(minLength: 34)

                waveform
                    .padding(.bottom, 22)

                Text("MeanVC 2")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .tracking(-1.2)
                    .foregroundStyle(.white)

                Text("REAL-TIME VOICE CONVERSION")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .tracking(2.4)
                    .foregroundStyle(.white.opacity(0.42))
                    .padding(.top, 7)

                Spacer(minLength: 34)

                referenceCard


                Button {
                    if isLive {
                        liveAudio.stop()
                        isLive = false
                    } else if let stream {
                        liveAudio.onError = { message in
                            Task { @MainActor in
                                modelError = "音声変換エラー: \(message)"
                                showingModelInfo = true
                                liveAudio.stop()
                                isLive = false
                            }
                        }
                        liveAudio.onRunningChanged = { running, status in
                            Task { @MainActor in
                                isLive = running
                                if !status.isEmpty {
                                    modelStatus = status
                                }
                            }
                        }
                        do {
                            try liveAudio.start(stream: stream)
                            isLive = true
                        } catch {
                            modelError = "マイクを開始できませんでした: \(error.localizedDescription)"
                            showingModelInfo = true
                        }
                    }
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: isLive ? "stop.fill" : "mic.fill")
                            .font(.system(size: 15, weight: .bold))
                        Text(isLive ? "変換を停止" : "マイク変換を開始")
                            .font(.system(size: 15, weight: .bold))
                        Spacer()
                        Text(isLive ? "LIVE" : (targetReady ? "READY" : "VOICE REQUIRED"))
                            .font(.system(size: 10, weight: .bold))
                            .tracking(1)
                    }
                    .foregroundStyle(.white)
                    .padding(17)
                    .background(
                        isLive ? Color.red.opacity(0.78) : accent.opacity(targetReady ? 0.9 : 0.25),
                        in: RoundedRectangle(cornerRadius: 18)
                    )
                }
                .buttonStyle(.plain)
                .disabled(!targetReady && !isLive)
                .padding(.top, 12)

                Button {
                    showingModelInfo = true
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: "waveform")
                            .font(.system(size: 15, weight: .semibold))
                        Text("変換エンジンの状態")
                            .font(.system(size: 15, weight: .semibold))
                        Spacer()
                        Text(engineReady ? "ONNX READY" : "準備中")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(accent)
                        Image(systemName: "chevron.right")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(.white.opacity(0.35))
                    }
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(17)
                    .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18))
                }
                .buttonStyle(.plain)
                .padding(.top, 12)

                Spacer(minLength: 26)

                Text(isLive ? "マイク変換中" : (targetReady ? "変換先の声を登録しました" : modelStatus))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.4))
                    .multilineTextAlignment(.center)
                    .padding(.bottom, 18)
            }
            .padding(.horizontal, 22)
        }
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.audio, .wav, .mpeg4Audio],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                referenceURL = url
                showingSpeakerConfirmation = true
            }
        }
        .alert("変換エンジン", isPresented: $showingModelInfo) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(modelError ?? "Fast-U2++ ASR、DiT、VocosをONNX Runtimeで実行します。変換モデルは端末内に保存し、推論はオフラインで行います。")
        }
        .confirmationDialog(
            "話者モデルの準備",
            isPresented: $showingSpeakerConfirmation,
            titleVisibility: .visible
        ) {
            Button("ダウンロードして登録") {
                if let url = referenceURL {
                    Task { await registerTargetVoice(from: url) }
                }
            }
            Button("あとで", role: .cancel) { }
        } message: {
            Text("初回のみ約387MBの話者モデルをダウンロードします。3秒以上のはっきりした音声を使い、本人の許可を得た音声だけを登録してください。")
        }
        .task {
            await prepareConversionEngine()
        }
        .onDisappear {
            liveAudio.stop()
            isLive = false
        }
    }

    private func modelsDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return base.appendingPathComponent("MeanVC2Models", isDirectory: true)
    }

    @MainActor
    private func prepareConversionEngine() async {
        guard stream == nil else { return }
        let isCISmokeTest = ProcessInfo.processInfo.arguments.contains("--meanvc2-ci-smoke")
        if isCISmokeTest {
            MeanVC2SmokeDiagnostics.write("started")
        }
        modelStatus = "変換モデルを準備しています"
        do {
            let directory = try modelsDirectory()
            let store = MeanVC2ModelStore(root: directory)
            _ = try await store.ensureConversionModels()
            if isCISmokeTest {
                MeanVC2SmokeDiagnostics.write("models-ready")
            }

            let (runner, smokeOutput, restoredTarget) = try await Task.detached(
                priority: .userInitiated
            ) { () throws -> (MeanVC2ONNXStream, [Float], Bool) in
                if isCISmokeTest {
                    MeanVC2SmokeDiagnostics.write("loading-sessions")
                }
                let runner = try MeanVC2ONNXStream(modelDirectory: directory)
                var output: [Float] = []

                // Synthetic zero audio and a zero speaker vector are strictly
                // a deterministic CI graph-execution fixture, never a user mode.
                if isCISmokeTest {
                    MeanVC2SmokeDiagnostics.write("sessions-ready")
                    runner.noiseProvider = { frameCount in
                        [Float](repeating: 0, count: frameCount * 80)
                    }
                    try runner.setTargetEmbedding([Float](repeating: 0, count: 256))
                    MeanVC2SmokeDiagnostics.write("conditioning-ready")
                    for index in 0..<8 {
                        MeanVC2SmokeDiagnostics.write("processing-chunk=\(index + 1)")
                        output.append(contentsOf: try runner.process(
                            [Float](repeating: 0, count: 2_560)
                        ))
                    }
                    guard !output.isEmpty, output.allSatisfy(\.isFinite) else {
                        throw MeanVC2ONNXError.invalidTensor(
                            "ONNX streaming smoke test returned no valid audio"
                        )
                    }
                    runner.noiseProvider = nil
                    MeanVC2SmokeDiagnostics.write(
                        "PASS samples=\(output.count)"
                    )
                }

                let savedEmbeddingURL = directory.appendingPathComponent("target-speaker.f32")
                var hasRestoredTarget = false
                if let savedData = try? Data(contentsOf: savedEmbeddingURL),
                   savedData.count == 256 * MemoryLayout<Float>.size {
                    let savedEmbedding = savedData.withUnsafeBytes { raw in
                        Array(raw.bindMemory(to: Float.self))
                    }
                    try runner.setTargetEmbedding(savedEmbedding)
                    hasRestoredTarget = true
                }
                return (runner, output, hasRestoredTarget)
            }.value

            if isCISmokeTest {
                #if DEBUG
                let documents = FileManager.default.urls(
                    for: .documentDirectory, in: .userDomainMask
                ).first!
                try? Data("PASS".utf8).write(
                    to: documents.appendingPathComponent("onnx-pipeline-smoke-test.ok"),
                    options: .atomic
                )
                let waveformData = smokeOutput.withUnsafeBufferPointer { buffer -> Data in
                    guard let base = buffer.baseAddress else { return Data() }
                    return Data(
                        bytes: base,
                        count: buffer.count * MemoryLayout<Float>.size
                    )
                }
                try? waveformData.write(
                    to: documents.appendingPathComponent("onnx-pipeline-smoke-output.f32"),
                    options: .atomic
                )
                #endif
            }

            stream = runner
            targetReady = restoredTarget
            engineReady = true
            modelStatus = restoredTarget
                ? "変換モデルと変換先の声を準備しました"
                : "変換モデル準備完了。変換先の声を登録してください"
            modelLogger.info("ONNX_RUNTIME_READY")
        } catch {
            if isCISmokeTest {
                MeanVC2SmokeDiagnostics.write("FAIL \(error.localizedDescription)")
            }
            modelError = "ONNXモデルの準備に失敗しました: \(error.localizedDescription)"
            modelStatus = "モデル準備に失敗"
            modelLogger.error("ONNX_RUNTIME_FAILED: \(error.localizedDescription)")
        }
    }

    @MainActor
    private func registerTargetVoice(from url: URL) async {
        guard engineReady else {
            modelStatus = "先に変換モデルの準備を完了してください"
            return
        }
        let didStartAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccess { url.stopAccessingSecurityScopedResource() }
        }

        if isLive {
            liveAudio.stop()
            isLive = false
        }
        targetReady = false
        modelStatus = "話者モデルを準備しています"
        do {
            let directory = try modelsDirectory()
            let store = MeanVC2ModelStore(root: directory)
            _ = try await store.ensureConversionModels(includeSpeakerEncoder: true) { file in
                Task { @MainActor in
                    modelStatus = "ダウンロード中: \(file)"
                }
            }
            modelStatus = "参照音声を解析しています"
            let speakerModel = directory.appendingPathComponent("speaker/speaker.int8.onnx")
            let embedding = try await Task.detached(priority: .userInitiated) {
                let waveform = try MeanVC2AudioFile.loadMono16k(from: url)
                let encoder = try MeanVC2SpeakerEncoder(modelURL: speakerModel)
                return try encoder.embedding(for: waveform)
            }.value
            try stream?.setTargetEmbedding(embedding)
            let embeddingData = embedding.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.size)
            }
            try embeddingData.write(
                to: directory.appendingPathComponent("target-speaker.f32"),
                options: .atomic
            )
            targetReady = true
            modelStatus = "変換先の声を登録しました"
        } catch {
            targetReady = false
            modelStatus = "話者登録に失敗"
            modelError = "話者登録に失敗しました: \(error.localizedDescription)"
            showingModelInfo = true
        }
    }

    private var header: some View {
        HStack {
            HStack(spacing: 8) {
                Circle()
                    .fill(accent)
                    .frame(width: 7, height: 7)
                    .shadow(color: accent.opacity(0.8), radius: 6)
                Text("OFFLINE VOICE AI")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .tracking(1.5)
                    .foregroundStyle(.white.opacity(0.65))
            }
            Spacer()
            Button {
                showingModelInfo = true
            } label: {
                Image(systemName: "info")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white.opacity(0.7))
                    .frame(width: 34, height: 34)
                    .background(Color.white.opacity(0.07), in: Circle())
            }
            .buttonStyle(.plain)
        }
    }

    private var waveform: some View {
        HStack(alignment: .center, spacing: 5) {
            ForEach(Array([18.0, 30, 43, 25, 54, 34, 20, 46, 60, 29, 42, 18, 34, 51, 24, 39, 17].enumerated()), id: \.offset) { item in
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [accent, Color(red: 0.34, green: 0.72, blue: 1.0)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 4, height: item.element)
            }
        }
        .frame(height: 64)
        .shadow(color: accent.opacity(0.22), radius: 18)
        .accessibilityLabel("音声波形")
    }

    private var referenceCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("VOICE REFERENCE")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .tracking(1.5)
                    .foregroundStyle(.white.opacity(0.45))
                Spacer()
                Text("ZERO-SHOT")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(1)
                    .foregroundStyle(accent)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(accent.opacity(0.13), in: Capsule())
            }

            Button {
                showingImporter = true
            } label: {
                HStack(spacing: 13) {
                    Image(systemName: referenceURL == nil ? "plus" : "waveform")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(accent)
                        .frame(width: 43, height: 43)
                        .background(accent.opacity(0.14), in: RoundedRectangle(cornerRadius: 13))

                    VStack(alignment: .leading, spacing: 4) {
                        Text(referenceURL?.lastPathComponent ?? (targetReady ? "保存済みの声" : "変換先の声を選択"))
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                            .lineLimit(1)
                        Text(referenceURL == nil ? "3秒以上の音声ファイルを読み込む" : "タップして別の音声に変更")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.white.opacity(0.42))
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white.opacity(0.3))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(17)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.085), Color.white.opacity(0.035)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 21)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 21)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        }
    }
}

#Preview {
    ContentView()
}
