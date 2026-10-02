#pragma once

#include "../board.h"
#include "../move.h"


struct MCTSNode {
    Board board;
    Move move;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    double visits;
    double score;
};

class MCTS {
    private:
        Mover mover;
    public:
        void search(MCTSNode* node, Board board);
        void backpropagate(MCTSNode* node, double result);
        double evaluate(Board& board);
        void deleteTree(MCTSNode* node);
        Move findBestMove(Board board, int iterations);
};