import Foundation

/// Tracks scheduled or queued audio frames without allowing unbounded latency.
struct MeanVC2AudioFrameBudget {
    let maximumFrames: Int
    private(set) var reservedFrames: Int = 0

    init(maximumFrames: Int) {
        precondition(maximumFrames > 0, "maximumFrames must be positive")
        self.maximumFrames = maximumFrames
    }

    /// Reserves frames if the new total stays within the configured limit.
    mutating func reserve(_ frameCount: Int) -> Bool {
        guard frameCount > 0,
              frameCount <= maximumFrames,
              reservedFrames <= maximumFrames - frameCount else {
            return false
        }
        reservedFrames += frameCount
        return true
    }

    /// Releases completed frames. Duplicate or late callbacks cannot underflow.
    mutating func release(_ frameCount: Int) {
        guard frameCount > 0 else { return }
        reservedFrames = max(0, reservedFrames - frameCount)
    }

    mutating func reset() {
        reservedFrames = 0
    }
}
