import AVFoundation
import Foundation

enum MeanVC2AudioFileError: Error {
    case unsupportedAudio
    case conversionFailed(String)
}

enum MeanVC2AudioFile {
    /// Reads an audio file, downmixes to mono and resamples to 16 kHz Float32.
    static func loadMono16k(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        guard sourceFormat.sampleRate > 0, file.length > 0 else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }

        guard let source = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }
        try file.read(into: source)
        guard source.frameLength > 0,
              let converter = AVAudioConverter(
                from: sourceFormat,
                to: AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: 16_000,
                    channels: 1,
                    interleaved: false
                )!
              ) else {
            throw MeanVC2AudioFileError.unsupportedAudio
        }

        let targetFormat = converter.outputFormat
        let estimatedFrames = AVAudioFrameCount(
            ceil(Double(source.frameLength) * targetFormat.sampleRate / sourceFormat.sampleRate)
        ) + 2_048
        guard let output = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: estimatedFrames
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
            return source
        }
        if let conversionError {
            throw MeanVC2AudioFileError.conversionFailed(conversionError.localizedDescription)
        }
        guard status != .error,
              output.frameLength > 0,
              let channel = output.floatChannelData?.pointee else {
            throw MeanVC2AudioFileError.conversionFailed("AVAudioConverter returned no samples")
        }
        return Array(UnsafeBufferPointer(
            start: channel,
            count: Int(output.frameLength)
        ))
    }
}
