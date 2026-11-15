import SwiftUI
import UniformTypeIdentifiers
import os

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

    private let accent = Color(red: 0.63, green: 0.48, blue: 1.0)
    private let modelLogger = Logger(subsystem: "jp.yanyantukebo.meanvc2", category: "Vocos")

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

                Text(targetReady ? "変換先の声を登録しました" : modelStatus)
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
            Text(vocosError ?? "VocosのCore ML推論とiOS側iSTFTを確認しました。Fast-U2++ ASR、話者特徴抽出、DiT、リアルタイム音声接続は開発中です。")
        }
        .task {
            guard let modelURL = Bundle.main.url(
                forResource: "MeanVC2VocosSpectrogram",
                withExtension: "mlmodelc"
            ) else {
                vocosError = "Vocosモデルがアプリ内に見つかりません。"
                return
            }
            do {
                let decoder = try VocosCoreMLDecoder(modelURL: modelURL)
                let sample = try decoder.decode(
                    mel: [Float](repeating: 0, count: 80 * 16),
                    frameCount: 16
                )
                guard sample.count == 2400, sample.allSatisfy(\.isFinite) else {
                    throw VocosDecoderError.invalidModelOutput
                }
                vocosDecoder = decoder
                vocosReady = true
                modelLogger.info("VOCOS_SMOKE_TEST_PASSED waveformSamples=\(sample.count)")
                #if DEBUG
                if let documents = FileManager.default.urls(
                    for: .documentDirectory,
                    in: .userDomainMask
                ).first {
                    let marker = documents.appendingPathComponent("vocos-smoke-test.ok")
                    try? Data("PASS".utf8).write(to: marker, options: .atomic)
                }
                #endif
            } catch {
                vocosError = "Vocos推論テストに失敗しました: \(error.localizedDescription)"
                modelLogger.error("VOCOS_SMOKE_TEST_FAILED: \(error.localizedDescription)")
            }
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
        modelStatus = "変換モデルをダウンロードしています"
        do {
            let directory = try modelsDirectory()
            let store = MeanVC2ModelStore(root: directory)
            _ = try await store.ensureConversionModels()
            let runner = try MeanVC2ONNXStream(modelDirectory: directory)
            try runner.setTargetEmbedding([Float](repeating: 0, count: 256))

            var smokeOutput: [Float] = []
            for _ in 0..<8 {
                smokeOutput.append(contentsOf: try runner.process(
                    [Float](repeating: 0, count: 2_560)
                ))
            }
            guard !smokeOutput.isEmpty, smokeOutput.allSatisfy(\\.isFinite) else {
                throw MeanVC2ONNXError.invalidTensor("ONNX streaming smoke test returned no valid audio")
            }
            stream = runner
            engineReady = true
            modelStatus = "変換モデル準備完了"
            modelLogger.info("ONNX_PIPELINE_SMOKE_TEST_PASSED samples=\\(smokeOutput.count)")
            #if DEBUG
            let documents = FileManager.default.urls(
                for: .documentDirectory, in: .userDomainMask
            ).first!
            try? Data("PASS".utf8).write(
                to: documents.appendingPathComponent("onnx-pipeline-smoke-test.ok"),
                options: .atomic
            )
            #endif
        } catch {
            modelError = "ONNXモデルの準備に失敗しました: \\(error.localizedDescription)"
            modelStatus = "モデル準備に失敗"
            modelLogger.error("ONNX_PIPELINE_SMOKE_TEST_FAILED: \\(error.localizedDescription)")
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
        modelStatus = "話者モデルを準備しています"
        do {
            let directory = try modelsDirectory()
            let store = MeanVC2ModelStore(root: directory)
            _ = try await store.ensureConversionModels(includeSpeakerEncoder: true) { file in
                Task { @MainActor in
                    modelStatus = "ダウンロード中: \\(file)"
                }
            }
            modelStatus = "参照音声を解析しています"
            let waveform = try MeanVC2AudioFile.loadMono16k(from: url)
            let speakerModel = directory.appendingPathComponent("speaker/speaker.int8.onnx")
            let encoder = try MeanVC2SpeakerEncoder(modelURL: speakerModel)
            let embedding = try encoder.embedding(for: waveform)
            try stream?.setTargetEmbedding(embedding)
            targetReady = true
            modelStatus = "変換先の声を登録しました"
        } catch {
            targetReady = false
            modelStatus = "話者登録に失敗"
            modelError = "話者登録に失敗しました: \\(error.localizedDescription)"
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
                        Text(referenceURL?.lastPathComponent ?? "変換先の声を選択")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                            .lineLimit(1)
                        Text(referenceURL == nil ? "音声ファイルを読み込む" : "タップして別の音声に変更")
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
