import Foundation

/// Writes CI-only simulator smoke diagnostics when an explicit launch
/// argument, simulator-injected environment flag, or one-shot Documents marker
/// is present. This is intentionally not gated by DEBUG.
struct MeanVC2SmokeDiagnostics: Sendable {
    static let launchArgument = "--meanvc2-ci-smoke"
    static let environmentKey = "MEANVC2_CI_SMOKE"
    static let statusFileName = "onnx-pipeline-smoke-test.status"
    static let successFileName = "onnx-pipeline-smoke-test.ok"
    static let outputFileName = "onnx-pipeline-smoke-output.f32"
    static let activationFileName = "meanvc2-ci-smoke.enable"

    enum DiagnosticsError: Error {
        case emptyWaveform
        case nonFiniteWaveform
    }

    let directory: URL

    static func fallbackDirectory(homeDirectory: URL) -> URL {
        homeDirectory.appendingPathComponent("Documents", isDirectory: true)
    }

    static func current(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        directory: URL? = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first,
        homeDirectory: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
    ) -> MeanVC2SmokeDiagnostics? {
        let argumentEnabled = arguments.contains(launchArgument)
        let environmentEnabled = environment[environmentKey] == "1"
        let resolvedDirectory = directory ?? fallbackDirectory(homeDirectory: homeDirectory)
        let activationURL = resolvedDirectory.appendingPathComponent(activationFileName)
        let fileEnabled = FileManager.default.fileExists(atPath: activationURL.path)
        NSLog(
            "[MeanVC2Smoke] bootstrap argument=%@ environment=%@ marker=%@ documents=%@",
            argumentEnabled ? "1" : "0",
            environmentEnabled ? "1" : "0",
            fileEnabled ? "1" : "0",
            resolvedDirectory.path
        )
        guard argumentEnabled || environmentEnabled || fileEnabled else {
            return nil
        }
        return MeanVC2SmokeDiagnostics(directory: resolvedDirectory)
    }

    func write(_ status: String) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let url = directory.appendingPathComponent(Self.statusFileName)
        try Data(status.utf8).write(to: url, options: .atomic)
        NSLog("[MeanVC2Smoke] %@", status)
    }

    /// Writes output first, then the success marker, and status last. CI
    /// therefore cannot observe PASS before both artifacts are durable.
    func writeOutput(_ samples: [Float]) throws {
        guard !samples.isEmpty else {
            throw DiagnosticsError.emptyWaveform
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw DiagnosticsError.nonFiniteWaveform
        }

        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let waveformData = samples.withUnsafeBufferPointer { buffer -> Data in
            guard let base = buffer.baseAddress else { return Data() }
            return Data(
                bytes: base,
                count: buffer.count * MemoryLayout<Float>.size
            )
        }
        try waveformData.write(
            to: directory.appendingPathComponent(Self.outputFileName),
            options: .atomic
        )
        try Data("PASS".utf8).write(
            to: directory.appendingPathComponent(Self.successFileName),
            options: .atomic
        )
        try write("PASS samples=\(samples.count)")
        // Consume the CI-only marker after a successful end-to-end smoke run.
        try? FileManager.default.removeItem(
            at: directory.appendingPathComponent(Self.activationFileName)
        )
    }
}
