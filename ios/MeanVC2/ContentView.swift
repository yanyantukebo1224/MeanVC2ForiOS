import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var referenceURL: URL?
    @State private var showingImporter = false
    @State private var showingModelInfo = false

    private let accent = Color(red: 0.63, green: 0.48, blue: 1.0)

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
                        Text("準備中")
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

                Text("モデルをiOS向けに最適化しています")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.35))
                    .padding(.bottom, 18)
            }
            .padding(.horizontal, 22)
        }
        .fileImporter(
            isPresented: $showingImporter,
            allowedContentTypes: [.audio, .wav, .mpeg4Audio],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result {
                referenceURL = urls.first
            }
        }
        .alert("変換エンジン", isPresented: $showingModelInfo) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("現在はiOSアプリのUIプロトタイプです。MeanVC2本体（ASR・話者特徴抽出・DiT・Vocos）のCore ML / Core Audio移植はこれから実装します。")
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
            ForEach(Array([18.0, 30, 43, 25, 54, 34, 20, 46, 60, 29, 42, 18, 34, 51, 24, 39, 17] .enumerated()), id: \.offset) { _, height in
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [accent, Color(red: 0.34, green: 0.72, blue: 1.0)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 4, height: height)
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
