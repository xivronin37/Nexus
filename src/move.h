#pragma once

#include "board.h"

enum class MoveType {
    Stone, Stick, Flip
};

struct Move {
    bool isPlayerOne;
    int row, col;
    bool orientH;
    MoveType type;
    int stoneChoices;
};

class Mover {
    public:
        void applyMove(Board& board, Move& move);
        bool isLegal(const Board& board, const Move& move);
        std::vector<Move> genMoves(const Board& board, bool isPlayerOne);
        uint64_t perft(const Board& board, int depth, bool isPlayerOne);
        int completeConnection(const Board& board, const Move& move);
        int floodFill(const Board& board, int r, int c);
        int decode(const StickRef& ref, int r, int c) const;
        int score(const Board& board, const Move& move);
        std::vector<int> stoneNeighbors(const Board& board, int row, int col) const;
};