import Foundation

/// Coalesces microphone-buffer drops into a single stream discontinuity signal.
/// The audio tap marks a gap; the serial inference queue consumes it and resets
/// model caches before accepting later samples.
final class MeanVC2InputDiscontinuity: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = false

    func markDroppedBuffer() {
        lock.lock()
        pending = true
        lock.unlock()
    }

    func consumePendingGap() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let hadGap = pending
        pending = false
        return hadGap
    }

    func clear() {
        lock.lock()
        pending = false
        lock.unlock()
    }
}
