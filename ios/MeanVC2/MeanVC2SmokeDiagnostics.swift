import Foundation

/// Writes CI-only simulator smoke diagnostics when the explicit launch
/// argument or simulator-injected environment flag is present. This is
/// intentionally not gated by DEBUG.
struct MeanVC2SmokeDiagnostics: Sendable {
    static let launchArgument = "--meanvc2-ci-smoke"
    static let environmentKey = "MEANVC2_CI_SMOKE"
    static let statusFileName = "onnx-pipeline-smoke-test.status"
    static let successFileName = "onnx-pipeline-smoke-test.ok"
    static let outputFileName = "onnx-pipeline-smoke-output.f32"

    enum DiagnosticsError: Error {
        case emptyWaveform
        case nonFiniteWaveform
    }

    let directory: URL

    static func current(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        directory: URL? = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first
    ) -> MeanVC2SmokeDiagnostics? {
        let argumentEnabled = arguments.contains(launchArgument)
        let environmentEnabled = environment[environmentKey] == "1"
        guard (argumentEnabled || environmentEnabled), let directory else {
            return nil
        }
        return MeanVC2SmokeDiagnostics(directory: directory)
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
    }
}
