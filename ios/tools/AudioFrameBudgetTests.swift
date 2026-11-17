import Foundation

@main
struct AudioFrameBudgetTests {
    static func main() {
        var checks = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            checks += 1
            guard condition() else {
                fatalError("Audio frame budget test failed: \(message)")
            }
        }

        var budget = MeanVC2AudioFrameBudget(maximumFrames: 10)
        check(budget.reserve(4), "first reservation should fit")
        check(budget.reservedFrames == 4, "reserved frame count should increase")
        check(!budget.reserve(7), "reservation beyond the limit should be rejected")
        check(budget.reservedFrames == 4, "rejected reservation must not mutate state")
        check(budget.reserve(6), "reservation that exactly fills the budget should fit")
        check(!budget.reserve(0), "zero-sized reservation should be rejected")
        check(!budget.reserve(-1), "negative reservation should be rejected")
        budget.release(3)
        check(budget.reservedFrames == 7, "release should subtract completed frames")
        budget.release(99)
        check(budget.reservedFrames == 0, "late completion must clamp at zero")
        budget.reset()
        check(budget.reservedFrames == 0, "reset should clear reservations")

        print("Audio frame budget tests passed (\(checks) assertions)")
    }
}
