import Foundation

/// Shared ONNX Runtime environment and session factory for the streaming
/// MeanVC2 graphs. Model sessions are CPU-first; XNNPACK/CoreML providers
/// are enabled only after device-specific benchmarks.
final class ONNXRuntimeEnvironment {
    static let shared: ORTEnv = {
        do {
            return try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        } catch {
            fatalError("Unable to initialize ONNX Runtime: \(error)")
        }
    }()

    static func makeSession(modelURL: URL, threads: Int = 2) throws -> ORTSession {
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(Int32(threads))
        try options.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
        return try ORTSession(
            env: shared,
            modelPath: modelURL.path,
            sessionOptions: options
        )
    }
}
