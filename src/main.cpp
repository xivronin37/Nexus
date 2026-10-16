
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "ai/infer.h"
#include "ncp.h"

int main(int argc, char* argv[]) {
    std::string modelPath;
    bool startingPlayerIsOne = true;
    bool policyPriorsEnabled = true;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--model" && argument + 1 < argc && modelPath.empty()) {
            modelPath = argv[++argument];
        } else if (option == "--start-player" && argument + 1 < argc) {
            const std::string player = argv[++argument];
            if (player == "1") {
                startingPlayerIsOne = true;
            } else if (player == "2") {
                startingPlayerIsOne = false;
            } else {
                std::cerr << "--start-player must be 1 or 2\n";
                return 2;
            }
        } else if (option == "--disable-policy-priors") {
            policyPriorsEnabled = false;
        } else {
            std::cerr << "Usage: Nexus-cli [--model <model.onnx>] "
                         "[--start-player 1|2] "
                         "[--disable-policy-priors]\n";
            return 2;
        }
    }

    try {
        std::unique_ptr<InferenceRuntime> runtime;
        std::unique_ptr<InferenceEngine> evaluator;
        NCP ncp;
        ncp.setStartingPlayer(startingPlayerIsOne);
        ncp.setPolicyPriorsEnabled(policyPriorsEnabled);

        if (!modelPath.empty()) {
            runtime = std::make_unique<InferenceRuntime>();
            evaluator = std::make_unique<InferenceEngine>();
            if (!evaluator->load(modelPath, *runtime)) {
                return 1;
            }
            ncp.setEvaluator(evaluator.get());
        }

        ncp.runNCP();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Engine startup failed: " << error.what() << '\n';
        return 1;
    }
}