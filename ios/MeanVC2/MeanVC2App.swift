import SwiftUI

@main
struct MeanVC2App: App {
    init() {
        // Prove that the simulator process received the CI launch signal
        // before SwiftUI view tasks or model loading begin.
        if let diagnostics = MeanVC2SmokeDiagnostics.current() {
            do {
                try diagnostics.write("app-init")
            } catch {
                NSLog("[MeanVC2Smoke] app-init diagnostic write failed: %@", error.localizedDescription)
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
