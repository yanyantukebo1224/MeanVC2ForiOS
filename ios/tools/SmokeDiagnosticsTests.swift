import Foundation

@main
struct SmokeDiagnosticsTests {
    static func main() throws {
        var checks = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            checks += 1
            guard condition() else {
                fatalError("Smoke diagnostics test failed: \(message)")
            }
        }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MeanVC2SmokeDiagnostics-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        check(
            MeanVC2SmokeDiagnostics.current(
                arguments: ["MeanVC2"], directory: root
            ) == nil,
            "smoke diagnostics must remain disabled without the launch flag"
        )

        let diagnostics = MeanVC2SmokeDiagnostics.current(
            arguments: ["MeanVC2", MeanVC2SmokeDiagnostics.launchArgument],
            directory: root
        )
        check(diagnostics != nil, "explicit launch flag should enable diagnostics")
        guard let diagnostics else { return }

        try diagnostics.write("app-init")
        let statusURL = root.appendingPathComponent(
            MeanVC2SmokeDiagnostics.statusFileName
        )
        check(
            try String(contentsOf: statusURL, encoding: .utf8) == "app-init",
            "status should be atomically written"
        )

        let samples: [Float] = [0.25, -0.5, 1.0]
        try diagnostics.writeOutput(samples)
        let outputURL = root.appendingPathComponent(
            MeanVC2SmokeDiagnostics.outputFileName
        )
        let outputData = try Data(contentsOf: outputURL)
        let decoded = outputData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        check(decoded == samples, "waveform artifact should preserve Float32 samples")
        check(
            try String(
                contentsOf: root.appendingPathComponent(
                    MeanVC2SmokeDiagnostics.successFileName
                ),
                encoding: .utf8
            ) == "PASS",
            "success marker should be written after waveform output"
        )
        check(
            try String(contentsOf: statusURL, encoding: .utf8) == "PASS samples=3",
            "PASS status should be the final status"
        )

        let invalidRoot = root.appendingPathComponent("invalid")
        let invalid = MeanVC2SmokeDiagnostics(directory: invalidRoot)
        do {
            try invalid.writeOutput([])
            fatalError("empty output must be rejected")
        } catch MeanVC2SmokeDiagnostics.DiagnosticsError.emptyWaveform {
            checks += 1
        }
        do {
            try invalid.writeOutput([0, .nan])
            fatalError("non-finite output must be rejected")
        } catch MeanVC2SmokeDiagnostics.DiagnosticsError.nonFiniteWaveform {
            checks += 1
        }
        check(
            !FileManager.default.fileExists(
                atPath: invalidRoot.appendingPathComponent(
                    MeanVC2SmokeDiagnostics.successFileName
                ).path
            ),
            "invalid output must never create a PASS marker"
        )

        print("Smoke diagnostics tests passed (\(checks) assertions)")
    }
}
