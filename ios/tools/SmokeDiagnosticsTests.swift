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
                arguments: ["MeanVC2"], environment: [:], directory: root
            ) == nil,
            "smoke diagnostics must remain disabled without an explicit signal"
        )

        let argumentDiagnostics = MeanVC2SmokeDiagnostics.current(
            arguments: ["MeanVC2", MeanVC2SmokeDiagnostics.launchArgument],
            environment: [:],
            directory: root
        )
        check(argumentDiagnostics != nil, "explicit launch argument should enable diagnostics")

        let environmentDiagnostics = MeanVC2SmokeDiagnostics.current(
            arguments: ["MeanVC2"],
            environment: [MeanVC2SmokeDiagnostics.environmentKey: "1"],
            directory: root
        )
        check(environmentDiagnostics != nil, "simulator-injected environment flag should enable diagnostics")

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let activationURL = root.appendingPathComponent(
            MeanVC2SmokeDiagnostics.activationFileName
        )
        try Data("enabled".utf8).write(to: activationURL)
        let fileDiagnostics = MeanVC2SmokeDiagnostics.current(
            arguments: ["MeanVC2"],
            environment: [:],
            directory: root
        )
        check(fileDiagnostics != nil, "one-shot app-container marker should enable diagnostics")
        if let fileDiagnostics {
            try fileDiagnostics.writeOutput([0.125])
        }
        check(
            !FileManager.default.fileExists(atPath: activationURL.path),
            "successful smoke output should consume the one-shot activation marker"
        )
        check(
            MeanVC2SmokeDiagnostics.current(
                arguments: ["MeanVC2"], environment: [:], directory: root
            ) == nil,
            "consumed activation marker must not enable a later normal launch"
        )

        for disabledValue in ["0", "true", ""] {
            check(
                MeanVC2SmokeDiagnostics.current(
                    arguments: ["MeanVC2"],
                    environment: [MeanVC2SmokeDiagnostics.environmentKey: disabledValue],
                    directory: root
                ) == nil,
                "environment value \(disabledValue.debugDescription) must not enable diagnostics"
            )
        }

        let fallbackHome = root.appendingPathComponent("fallback-home", isDirectory: true)
        let expectedFallback = fallbackHome.appendingPathComponent("Documents", isDirectory: true)
        check(
            MeanVC2SmokeDiagnostics.fallbackDirectory(homeDirectory: fallbackHome) == expectedFallback,
            "missing Foundation document URL should fall back to the app container Documents path"
        )
        let fallbackDiagnostics = MeanVC2SmokeDiagnostics.current(
            arguments: ["MeanVC2"],
            environment: [MeanVC2SmokeDiagnostics.environmentKey: "1"],
            directory: nil,
            homeDirectory: fallbackHome
        )
        check(fallbackDiagnostics?.directory == expectedFallback,
              "explicit CI signal should remain enabled when Foundation omits the document URL")
        try fallbackDiagnostics?.write("fallback-init")
        check(
            FileManager.default.fileExists(
                atPath: expectedFallback.appendingPathComponent(
                    MeanVC2SmokeDiagnostics.statusFileName
                ).path
            ),
            "fallback Documents directory should accept a bootstrap status file"
        )

        guard let diagnostics = environmentDiagnostics else { return }
        try diagnostics.write("app-init")
        let statusURL = root.appendingPathComponent(
            MeanVC2SmokeDiagnostics.statusFileName
        )
        let initialStatus = try String(contentsOf: statusURL, encoding: .utf8)
        check(initialStatus == "app-init", "status should be atomically written")

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
        let successMarker = try String(
            contentsOf: root.appendingPathComponent(
                MeanVC2SmokeDiagnostics.successFileName
            ),
            encoding: .utf8
        )
        check(successMarker == "PASS", "success marker should be written after waveform output")
        let finalStatus = try String(contentsOf: statusURL, encoding: .utf8)
        check(finalStatus == "PASS samples=3", "PASS status should be the final status")

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
