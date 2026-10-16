#pragma once

#include <cstddef>
#include <mutex>

class InferenceEngine;

struct MatchResults {
    double llr = 0.0;
    std::size_t gamesPlayed = 0.0;
    std::size_t modelAWins = 0;
    std::size_t draws = 0;
    std::size_t modelBWins = 0;
    std::size_t modelAPlayerOneWins = 0;
    std::size_t modelAPlayerTwoWins = 0;
    std::size_t playerOneWins = 0;
    std::size_t playerTwoWins = 0;
    std::size_t startingPlayerWins = 0;
    std::size_t nonStartingPlayerWins = 0;
    std::size_t extraStickHolderWins = 0;
    std::size_t otherResourcePlayerWins = 0;
    std::size_t playerOneWinsWithExtraStick = 0;
    std::size_t playerTwoWinsWithExtraStick = 0;
    std::size_t playerOneWinsWithoutExtraStick = 0;
    std::size_t playerTwoWinsWithoutExtraStick = 0;
    int promotion = 0;
};

class MatchManager {
    private:
        double p0 = 0.50;
        double p1 = 0.55;
        double alpha = 0.05;
        double beta = 0.05;
        std::mutex coutMutex;
    public:
        MatchResults runGame(InferenceEngine* modelA,
            InferenceEngine* modelB,
            int gameIndex,
            int iterationsPerMove,
            std::size_t maxPlies,
            int maxGames = 1,
            bool alternateStartingPlayer = false,
            bool alternateResourceAdvantage = false,
            bool reverseResourceAdvantage = false,
            bool policyPriorsAEnabled = true,
            bool policyPriorsBEnabled = true,
            double priorScaleA = 1.0,
            double priorScaleB = 1.0
        );
        MatchResults run(
            InferenceEngine* modelA,
            InferenceEngine* modelB,
            std::size_t games,
            int iterationsPerMove,
            std::size_t maxPlies,
            bool alternateStartingPlayer = false,
            bool alternateResourceAdvantage = false,
            bool reverseResourceAdvantage = false,
            bool policyPriorsAEnabled = true,
            bool policyPriorsBEnabled = true,
            double priorScaleA = 1.0,
            double priorScaleB = 1.0
        );

        MatchResults runSPRT(
            InferenceEngine* modelA,
            InferenceEngine* modelB,
            std::size_t maxGames,
            int iterationsPerMove,
            std::size_t maxPlies,
            bool policyPriorsAEnabled = true,
            bool policyPriorsBEnabled = true,
            double priorScaleA = 1.0,
            double priorScaleB = 1.0
        );

        MatchResults runSPRTParallel(
            InferenceEngine* modelA,
            InferenceEngine* modelB,
            std::size_t maxGames,
            int iterationsPerMove,
            std::size_t maxPlies,
            bool policyPriorsAEnabled = true,
            bool policyPriorsBEnabled = true,
            double priorScaleA = 1.0,
            double priorScaleB = 1.0
            );
    
};
