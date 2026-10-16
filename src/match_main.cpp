#include <cctype>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include "board.h"
#include "ai/infer.h"
#include "ai/match_manager.h"
#include "move.h"

namespace {
std::size_t parsePositiveSize(const char* text, const char* name) {
    const std::string value(text);
    if (value.empty()) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer");
    }

    for (char character : value) {
        if (!std::isdigit(static_cast<unsigned char>(character))) {
            throw std::invalid_argument(
                std::string(name) + " must be a positive integer");
        }
    }

    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

Board swapPlayers(const Board& board) {
    Board swapped = board;
    swapped.isPlayerOne = !board.isPlayerOne;
    std::swap(swapped.stonesOne, swapped.stonesTwo);
    std::swap(swapped.sticksOne, swapped.sticksTwo);
    std::swap(swapped.ptsOne, swapped.ptsTwo);
    swapped.lastFlippedIsOne = !board.lastFlippedIsOne;

    for (std::size_t i = 0; i < board.stoneSet.size(); ++i) {
        swapped.stoneOwner.set(
            i, board.stoneSet.test(i) && !board.stoneOwner.test(i));
    }
    for (std::size_t i = 0; i < board.hStickSet.size(); ++i) {
        swapped.hStickOwner.set(
            i, board.hStickSet.test(i) && !board.hStickOwner.test(i));
        swapped.vStickOwner.set(
            i, board.vStickSet.test(i) && !board.vStickOwner.test(i));
    }
    return swapped;
}

int runSymmetryDiagnostic(
    const std::string& modelPath,
    std::size_t positions,
    std::size_t maxRandomPlies) {
    InferenceRuntime runtime;
    InferenceEngine model;
    if (!model.load(modelPath, runtime)) {
        return 1;
    }

    Mover mover;
    std::mt19937 random(0x4E657875);
    double totalAbsoluteDifference = 0.0;
    std::size_t substantiallyDifferentPairs = 0;

    std::cout << "Prediction symmetry test (expected side-swapped positions "
                 "to have similar values)\n";
    for (std::size_t sample = 0; sample < positions; ++sample) {
        Board board;
        std::uniform_int_distribution<std::size_t> plyDistribution(
            1, maxRandomPlies);
        const std::size_t targetPlies = plyDistribution(random);

        for (std::size_t ply = 0; ply < targetPlies; ++ply) {
            if (mover.isGameOver(board) != 0) {
                break;
            }
            std::vector<Move> moves =
                mover.genMoves(board, board.isPlayerOne);
            if (moves.empty()) {
                break;
            }
            std::uniform_int_distribution<std::size_t> moveDistribution(
                0, moves.size() - 1);
            Move move = moves[moveDistribution(random)];
            mover.applyMove(board, move);
        }

        const float originalValue = model.predict(board);
        const float swappedValue = model.predict(swapPlayers(board));
        const float difference = originalValue - swappedValue;
        totalAbsoluteDifference += std::abs(difference);
        if (std::abs(difference) > 0.1f) {
            ++substantiallyDifferentPairs;
        }

        std::cout << "Position " << (sample + 1)
                  << ": value=" << originalValue
                  << ", swapped=" << swappedValue
                  << ", difference=" << difference << '\n';
    }

    std::cout << "Mean absolute value difference: "
              << totalAbsoluteDifference / static_cast<double>(positions)
              << '\n'
              << "Pairs with difference > 0.1: "
              << substantiallyDifferentPairs << '/' << positions << '\n';
    return 0;
}
}

int main(int argc, char* argv[]) {
    if (argc >= 3 && std::string(argv[2]) == "symmetry") {
        if (argc > 5) {
            std::cerr << "Usage: Nexus-match <model.onnx> symmetry"
                         " [positions=32] [max-random-plies=40]\n";
            return 2;
        }
        try {
            const std::size_t positions =
                argc > 3 ? parsePositiveSize(argv[3], "positions") : 32;
            const std::size_t maxRandomPlies =
                argc > 4
                    ? parsePositiveSize(argv[4], "max-random-plies")
                    : 40;
            return runSymmetryDiagnostic(
                argv[1], positions, maxRandomPlies);
        } catch (const std::exception& error) {
            std::cerr << "Symmetry diagnostic failed: "
                      << error.what() << '\n';
            return 1;
        }
    }

    if (argc < 3 || argc > 11) {
        std::cerr << "Usage: Nexus-match <model-A.onnx|heuristic> "
                     "<model-B.onnx|heuristic>"
                     " [games=20] [iterations-per-move=100] [max-plies=150]"
                     " [alternate-start|balanced|equal-resources|"
                     "normal-resources|reversed-resources]"
                     " [--no-policy-A] [--no-policy-B] [priorScaleA=1.0] [priorScaleB=1.0]\n"
                     "       Nexus-match <model.onnx> symmetry"
                     " [positions=32] [max-random-plies=40]\n";
        return 2;
    }

    try {
        std::string matchMode;
        bool policyPriorsAEnabled = true;
        bool policyPriorsBEnabled = true;
        double priorScaleA = 1.0;
        double priorScaleB = 1.0;
        int scaleCounter = 0;
        for (int argument = 6; argument < argc; ++argument) {
            const std::string option = argv[argument];
            if (option == "alternate-start" || option == "balanced" ||
                option == "equal-resources" ||
                option == "normal-resources" ||
                option == "reversed-resources") {
                if (!matchMode.empty()) {
                    throw std::invalid_argument(
                        "Specify at most one match/resource mode");
                }
                matchMode = option;
            } else if (option == "--no-policy-A") {
                policyPriorsAEnabled = false;
            } else if (option == "--no-policy-B") {
                policyPriorsBEnabled = false;
            } else {
                bool conversionSuccess = false;
                double value = 0.0;
                
                try {
                    size_t pos = 0;
                    value = std::stod(option, &pos);
                    if (pos == option.size()) {
                        conversionSuccess = true;
                    }
                } catch (...) {
                    // do nothing
                }

                if (conversionSuccess) {
                    if (value <= 0.0 || !std::isfinite(value)) {
                        throw std::invalid_argument(
                            "Prior scale values must be finite and greater than 0: " + option);
                    }

                    if (scaleCounter == 0) {
                        priorScaleA = value;
                    } else if (scaleCounter == 1) {
                        priorScaleB = value;
                    } else {
                        throw std::invalid_argument(
                            "Specify at most two prior scale numbers");
                    }
                    ++scaleCounter;
                } else {
                    throw std::invalid_argument(
                        "Unknown match option: " + option);
                }
            }
        }
        const std::size_t games =
            argc > 3 ? parsePositiveSize(argv[3], "games") : 20;
        const std::size_t iterations =
            argc > 4 ? parsePositiveSize(argv[4], "iterations") : 100;
        const std::size_t maxPlies =
            argc > 5 ? parsePositiveSize(argv[5], "max-plies") : 150;
        const bool alternateStartingPlayer =
            matchMode == "alternate-start" || matchMode == "balanced" ||
            matchMode == "equal-resources" ||
            matchMode == "normal-resources" ||
            matchMode == "reversed-resources";
        const bool alternateResourceAdvantage = matchMode == "balanced";
        const bool equalizeResources = matchMode == "equal-resources";
        const bool reverseResourceAdvantage =
            matchMode == "reversed-resources";
        if (alternateResourceAdvantage && games % 8 != 0) {
            throw std::invalid_argument(
                "Balanced mode requires a game count divisible by 8");
        }
        if (equalizeResources && games % 4 != 0) {
            throw std::invalid_argument(
                "Equal-resources mode requires a game count divisible by 4");
        }
        if (iterations > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "iterations exceeds the supported integer range");
        }

        InferenceRuntime runtime;
        InferenceEngine modelA;
        InferenceEngine modelB;
        const bool useHeuristicA = std::string(argv[1]) == "heuristic";
        const bool useHeuristicB = std::string(argv[2]) == "heuristic";
        std::cout << "Policy priors: Model A "
                  << (policyPriorsAEnabled ? "enabled" : "disabled")
                  << ", Model B "
                  << (policyPriorsBEnabled ? "enabled" : "disabled") << '\n';

        if (useHeuristicA) {
            std::cout << "Model A: built-in heuristic evaluator\n";
        } else {
            std::cout << "Loading Model A: " << argv[1] << '\n';
            if (!modelA.load(argv[1], runtime)) {
                return 1;
            }
        }
        if (!useHeuristicB) {
            std::cout << "Loading Model B: " << argv[2] << '\n';
            if (!modelB.load(argv[2], runtime)) {
                return 1;
            }
        } else {
            std::cout << "Model B: built-in heuristic evaluator\n";
        }

        MatchManager manager;
        const MatchResults results = manager.runSPRTParallel(
            useHeuristicA ? nullptr : &modelA,
            useHeuristicB ? nullptr : &modelB,
            games,
            static_cast<int>(iterations),
            maxPlies,
            policyPriorsAEnabled,
            policyPriorsBEnabled,
            priorScaleA,
            priorScaleB
        );

        std::cout << "\nFinal results\n"
                  << "Model A wins: " << results.modelAWins << '\n'
                  << "  as Player 1: " << results.modelAPlayerOneWins << '\n'
                  << "  as Player 2: " << results.modelAPlayerTwoWins << '\n'
                  << "Player 1 wins: " << results.playerOneWins << '\n'
                  << "Player 2 wins: " << results.playerTwoWins << '\n'
                  << "Starting-player wins: " << results.startingPlayerWins << '\n'
                  << "Non-starting-player wins: "
                  << results.nonStartingPlayerWins << '\n';
        if (alternateResourceAdvantage) {
            std::cout << "Extra-stick-holder wins: "
                      << results.extraStickHolderWins << '\n'
                      << "Other player's wins: "
                      << results.otherResourcePlayerWins << '\n'
                      << "P1 wins with extra stick: "
                      << results.playerOneWinsWithExtraStick << '\n'
                      << "P2 wins with extra stick: "
                      << results.playerTwoWinsWithExtraStick << '\n'
                      << "P1 wins without extra stick: "
                      << results.playerOneWinsWithoutExtraStick << '\n'
                      << "P2 wins without extra stick: "
                      << results.playerTwoWinsWithoutExtraStick << '\n';
        }
        std::cout << "Draws: " << results.draws << '\n'
        << (useHeuristicB ? "Heuristic B wins: " : "Model B wins: ")
                  << results.modelBWins << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Match failed: " << error.what() << '\n';
        return 1;
    }
}
