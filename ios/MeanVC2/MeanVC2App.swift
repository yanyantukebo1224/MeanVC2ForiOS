import SwiftUI

@main
struct MeanVC2App: App {
    init() {
        // Prove that the simulator process received the CI launch argument
        // before SwiftUI view tasks or model loading begin.
        if let diagnostics = MeanVC2SmokeDiagnostics.current() {
            try? diagnostics.write("app-init")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
