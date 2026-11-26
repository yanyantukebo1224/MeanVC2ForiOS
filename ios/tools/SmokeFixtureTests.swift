import Foundation

@main
struct SmokeFixtureTests {
    static func main() throws {
        var assertions = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            assertions += 1
            guard condition() else {
                fatalError("Smoke fixture test failed: \(message)")
            }
        }

        let chunks = (0..<MeanVC2CISmokeFixture.chunkCount).map {
            MeanVC2CISmokeFixture.waveformChunk(index: $0)
        }
        let waveform = chunks.flatMap { $0 }
        let speaker = MeanVC2CISmokeFixture.speakerEmbedding()
        let noise = MeanVC2CISmokeFixture.noise(frameCount: 16)
        let speakerNorm = sqrt(
            speaker.reduce(0.0) { $0 + Double($1) * Double($1) }
        )

        check(chunks.count == 8, "fixture must contain eight chunks")
        check(chunks.allSatisfy { $0.count == 2_560 }, "each chunk must contain 160 ms")
        check(waveform.allSatisfy(\.isFinite), "waveform must be finite")
        check(waveform.contains { abs($0) > 0.01 }, "waveform must not be silent")
        check(chunks[0] != chunks[1], "adjacent chunks must not repeat")
        check(speaker.count == 256, "speaker conditioning must contain 256 values")
        check(speaker.allSatisfy(\.isFinite), "speaker conditioning must be finite")
        check(abs(speakerNorm - 1.0) < 1e-6, "speaker conditioning must have unit norm")
        check(noise.count == 16 * 80, "noise must use [frames, 80] layout")
        check(noise.contains { $0 != 0 }, "diffusion noise must be non-zero")

        guard CommandLine.arguments.count == 2 else {
            fatalError("Usage: SmokeFixtureTests <output-directory>")
        }
        let directory = URL(
            fileURLWithPath: CommandLine.arguments[1],
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        func write(_ values: [Float], name: String) throws {
            let data = values.withUnsafeBufferPointer { buffer -> Data in
                guard let base = buffer.baseAddress else { return Data() }
                return Data(
                    bytes: base,
                    count: buffer.count * MemoryLayout<Float>.stride
                )
            }
            try data.write(
                to: directory.appendingPathComponent(name),
                options: .atomic
            )
        }

        try write(waveform, name: "waveform.f32")
        try write(speaker, name: "speaker.f32")
        try write(noise, name: "noise.f32")
        print("Smoke fixture tests passed (\(assertions) assertions)")
    }
}
