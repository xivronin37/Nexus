#pragma once

#include "../board.h"
#include "../move.h"
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

class InferenceEngine;

struct MCTSNode;

struct PolicySearchResult {
    Move move;
    std::vector<std::pair<std::size_t, double>> visitDistribution;
};

struct MoveSlot {
    Move move;
    double prior;
    MCTSNode* child = nullptr;  // nullptr until this move gets expanded
};


struct MCTSNode {
    Board board;
    Move move;
    MCTSNode* parent;
    std::vector<MoveSlot> slots;
    double visits;
    double score;
    bool slotsBuilt = false;  
};

class MCTS {
    private:
        Mover mover;
        static inline std::mt19937 gen{std::random_device{}()};
    public:
        InferenceEngine* evaluator = nullptr;
        explicit MCTS(InferenceEngine* evaluator = nullptr)
            : evaluator(evaluator) {}
        void setEvaluator(InferenceEngine* value) { evaluator = value; }
        void setPolicyPriorsEnabled(bool enabled) {
            policyPriorsEnabled = enabled;
        }
        double priorScale = 0;
        int maxDepth = 0;
        void search(MCTSNode* node, Board board, int depth = 0);
        void backpropagate(MCTSNode* node, double result);
        double evaluate(Board& board);
        void deleteTree(MCTSNode* node);
        Move findBestMove(Board board, int iterations);
        PolicySearchResult findBestMoveWithVisitDistribution(
            Board board,
            int iterations,
            double temperature);
        Move stochasticFindBestMove(Board board, int iterations, double temperature);
        Move pickMove(Board board, int idx);
    private:
        bool policyPriorsEnabled = true;
};

std::size_t policyActionIndex(const Move& move);