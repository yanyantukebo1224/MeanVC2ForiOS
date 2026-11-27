import Foundation

@main
struct InputDiscontinuityTests {
    static func main() {
        let state = MeanVC2InputDiscontinuity()
        var assertions = 0

        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            assertions += 1
            guard condition() else {
                fatalError("Input discontinuity test failed: \\(message)")
            }
        }

        check(!state.consumePendingGap(), "new state must not report a gap")

        state.markDroppedBuffer()
        check(state.consumePendingGap(), "one dropped buffer must report a gap")
        check(!state.consumePendingGap(), "consuming a gap must clear it")

        state.markDroppedBuffer()
        state.markDroppedBuffer()
        state.markDroppedBuffer()
        check(state.consumePendingGap(), "multiple drops must coalesce into one reset")
        check(!state.consumePendingGap(), "coalesced gap must be consumed once")

        state.markDroppedBuffer()
        state.clear()
        check(!state.consumePendingGap(), "clear must discard stale gap state")

        DispatchQueue.concurrentPerform(iterations: 256) { _ in
            state.markDroppedBuffer()
        }
        check(state.consumePendingGap(), "concurrent drops must not lose the gap")
        check(!state.consumePendingGap(), "concurrent gap must be consumed once")

        print("Input discontinuity tests passed (\\(assertions) assertions)")
    }
}
