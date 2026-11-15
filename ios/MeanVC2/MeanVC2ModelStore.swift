import CryptoKit
import Foundation

enum MeanVC2ModelStoreError: Error {
    case badHTTPStatus(Int)
    case sizeMismatch(String)
    case checksumMismatch(String)
}

/// Downloads the published 120ms_40ms ONNX graph set from a pinned Hugging
/// Face revision. Every file is SHA-256 checked before it is made available
/// to ONNX Runtime. Conversion graphs are Apache-2.0; the optional speaker
/// encoder has a separate CC BY-SA 3.0 license.
actor MeanVC2ModelStore {
    static let revision = "2db97430c6e315a819daa30e255e14615205a3d1"
    static let repository = "https://huggingface.co/kiarina/MeanVC2-ONNX/resolve/\(revision)"

    struct Asset: Sendable {
        let name: String
        let bytes: Int
        let sha256: String
    }

    static let conversionAssets: [Asset] = [
        Asset(name: "fbank.onnx", bytes: 908339,
              sha256: "ecbe193170abe98a8c0ba9299e8fe21b418145aca924e351c667b9257c3e28c8"),
        Asset(name: "asr.int8.onnx", bytes: 28655587,
              sha256: "553869e3159af4497582072e66f09c16032a3b98ac2bad5c195b7443627616be"),
        Asset(name: "gtm.onnx", bytes: 17441893,
              sha256: "f03ff1e9da3c373445f7518fa36fc99980fe4326447235fa89ea1923f57ee224"),
        Asset(name: "dit.int8.onnx", bytes: 13849138,
              sha256: "52ac682ed1790b196333d1dbf99fa2806b6c52076c85e438c9dede968572675a"),
        Asset(name: "vocos.int8.onnx", bytes: 12998461,
              sha256: "a0dbd4c83708f6e787ceebf1c1b0833398dbdea6af770887d27b19119bba2e16")
    ]

    static let speakerAsset = Asset(
        name: "speaker/speaker.int8.onnx",
        bytes: 387106790,
        sha256: "ccd31b45eb2162937b94921d6e64fdd2cf5bf77193ad6f738e85f06a914a95ca"
    )

    private let root: URL
    private let session: URLSession

    init(root: URL, session: URLSession = .shared) {
        self.root = root
        self.session = session
    }

    func ensureConversionModels(
        includeSpeakerEncoder: Bool = false,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> URL {
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true
        )
        let assets = Self.conversionAssets + (includeSpeakerEncoder ? [Self.speakerAsset] : [])
        for asset in assets {
            let destination = root.appendingPathComponent(asset.name)
            if FileManager.default.fileExists(atPath: destination.path),
               try isValid(destination, asset: asset) {
                continue
            }
            progress?(asset.name)
            try await download(asset, to: destination)
        }
        return root
    }

    private func download(_ asset: Asset, to destination: URL) async throws {
        let url = URL(string: "\(Self.repository)/\(asset.name)?download=true")!
        let (temporary, response) = try await session.download(from: url)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw MeanVC2ModelStoreError.badHTTPStatus(status)
        }
        let actualSize = try FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber
        guard actualSize?.intValue == asset.bytes else {
            throw MeanVC2ModelStoreError.sizeMismatch(asset.name)
        }
        guard try sha256(of: temporary) == asset.sha256 else {
            throw MeanVC2ModelStoreError.checksumMismatch(asset.name)
        }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let staging = destination.appendingPathExtension("download")
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.copyItem(at: temporary, to: staging)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
    }

    private func isValid(_ url: URL, asset: Asset) async throws -> Bool {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard (attributes[.size] as? NSNumber)?.intValue == asset.bytes else {
            return false
        }
        return try sha256(of: url) == asset.sha256
    }

    private func sha256(of url: URL) throws -> String {
        let input = try InputStream(url: url).unwrap()
        input.open()
        defer { input.close() }
        var hasher = SHA256()
        let bufferSize = 1024 * 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while true {
            let count = input.read(buffer, maxLength: bufferSize)
            if count < 0 { throw input.streamError ?? CocoaError(.fileReadUnknown) }
            if count == 0 { break }
            hasher.update(data: Data(bytes: buffer, count: count))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private extension Optional where Wrapped == InputStream {
    func unwrap() throws -> InputStream {
        guard let value = self else { throw CocoaError(.fileReadUnknown) }
        return value
    }
}
