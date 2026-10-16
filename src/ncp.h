#pragma once

#include "board.h"
#include "move.h"
#include "ai/mcts.h"

class NCP {
    private:
        Board start;
        Mover mover;
        MCTS ai;
    public:
        void setEvaluator(InferenceEngine* evaluator) {
            ai.setEvaluator(evaluator);
        }
        void setPolicyPriorsEnabled(bool enabled) {
            ai.setPolicyPriorsEnabled(enabled);
        }
        void setStartingPlayer(bool isPlayerOne) {
            start.isPlayerOne = isPlayerOne;
        }
        void runNCP();
};