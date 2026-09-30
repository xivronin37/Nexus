
#include <iostream>
#include <deque>

#include "board.h"
#include "move.h"

void Mover::applyMove(Board& board, Move& move) {
    switch(move.type) {
        case MoveType::Stone: {
            int total = score(board, move);
            int connect = completeConnection(board, move);
            int index = stoneIndex(move.row, move.col);
            board.stoneSet.set(index);
            board.stoneOwner.set(index, move.isPlayerOne);
            move.isPlayerOne ? board.stonesOne-- : board.stonesTwo--;

            int& pts = move.isPlayerOne ? board.ptsOne : board.ptsTwo;
            int& stones = move.isPlayerOne ? board.stonesOne : board.stonesTwo;
            int& sticks = move.isPlayerOne ? board.sticksOne : board.sticksTwo;
            pts += total;
            stones += move.stoneChoices;
            sticks += (connect - move.stoneChoices);
            board.isPlayerOne = !board.isPlayerOne;
            break;
            }
        case MoveType::Stick: {
            int index = move.orientH ? hStickIndex(move.row, move.col) : vStickIndex(move.row, move.col);
            if (move.orientH) {
                int total = score(board, move);
                int connect = completeConnection(board, move);
                board.hStickSet.set(index);
                board.hStickOwner.set(index, move.isPlayerOne);
                move.isPlayerOne ? board.sticksOne-- : board.sticksTwo--;

                int& pts = move.isPlayerOne ? board.ptsOne : board.ptsTwo;
                int& stones = move.isPlayerOne ? board.stonesOne : board.stonesTwo;
                int& sticks = move.isPlayerOne ? board.sticksOne : board.sticksTwo;
                pts += total;
                stones += move.stoneChoices;
                sticks += (connect - move.stoneChoices);
                board.isPlayerOne = !board.isPlayerOne;
                break;
            } else {
                int total = score(board, move);
                int connect = completeConnection(board, move);
                board.vStickSet.set(index);
                board.vStickOwner.set(index, move.isPlayerOne);
                move.isPlayerOne ? board.sticksOne-- : board.sticksTwo--;

                int& pts = move.isPlayerOne ? board.ptsOne : board.ptsTwo;
                int& stones = move.isPlayerOne ? board.stonesOne : board.stonesTwo;
                int& sticks = move.isPlayerOne ? board.sticksOne : board.sticksTwo;
                pts += total;
                stones += move.stoneChoices;
                sticks += (connect - move.stoneChoices);
                board.isPlayerOne = !board.isPlayerOne;
                break;
            }
        }
        case MoveType::Flip: {
            int index = stoneIndex(move.row, move.col);
            board.stoneOwner.flip(index);
            std::vector<StickRef> ref = board.findConnections(move.row, move.col);
            for (StickRef r : ref) {
                r.isH ? board.hStickOwner.flip(r.index) : board.vStickOwner.flip(r.index);
            }
            board.isPlayerOne = !board.isPlayerOne;
            break;
        }
    }
}

bool Mover::isLegal(const Board& board, const Move& move) {
    switch(move.type) {
        default: return false;
        case MoveType::Stone:{
            if (move.isPlayerOne) {
                if (board.stonesOne == 0) {
                    return false;
                }
            } else {
                if (board.stonesTwo == 0) {
                        return false;
                    }
            }

            int index = stoneIndex(move.row, move.col);
            if ((0 <= move.row && move.row <= 11) && (0 <= move.col && move.col <= 11)) {
                bool exists = board.stoneSet.test(index);
                return !exists;
            }

            if (move.stoneChoices < 0 || move.stoneChoices > completeConnection(board, move)) {
                return false;
            }
        }
        case MoveType::Stick: {
            if (move.isPlayerOne) {
                if (board.sticksOne == 0) {
                    return false;
                }
            } else {
                if (board.sticksTwo == 0) {
                        return false;
                    }
            }
            int index = move.orientH ? hStickIndex(move.row, move.col) : vStickIndex(move.row, move.col);
            int left = stoneIndex(move.row, move.col);
            int right = stoneIndex(move.row, move.col + 1);
            int up = stoneIndex(move.row, move.col);
            int down = stoneIndex(move.row + 1, move.col);
            bool invalid, ends;
            if (move.orientH) {
                if ((0 <= move.row && move.row <= 11) && (0 <= move.col && move.col <= 10)) {
                    invalid = board.hStickSet.test(index);
                    ends = board.stoneSet.test(left) || board.stoneSet.test(right);
                    return !invalid && ends;
                }
                return false;
            } else {
                if ((0 <= move.row && move.row <= 10) && (0 <= move.col && move.col <= 11)) {
                    invalid = board.vStickSet.test(index);
                    ends = board.stoneSet.test(up) || board.stoneSet.test(down);
                    return !invalid && ends;
                }
                return false;
            }

            if (move.stoneChoices < 0 || move.stoneChoices > completeConnection(board, move)) {
                return false;
            }
        }
        case MoveType::Flip: {
            int index = stoneIndex(move.row, move.col);
            bool exists;
            if ((0 <= move.row && move.row <= 11) && (0 <= move.col && move.col <= 11)) {
                exists = board.stoneSet.test(index);
            } else {
                return false;
            }
            bool owner = board.stoneOwner.test(index);

            if (move.stoneChoices < 0 || move.stoneChoices > completeConnection(board, move)) {
                return false;
            }

            return exists && owner != move.isPlayerOne && board.degree(move.row, move.col) == 3;

        }
    }

    return false;
}

std::vector<Move> Mover::genMoves(const Board& board, bool isPlayerOne) {
    std::vector<Move> gen;
    gen.reserve(200);
    for (int i = 0; i < 144; i++) {
        int row = i / 12;
        int col = i % 12;
        
        Move stone{isPlayerOne, row, col, false, MoveType::Stone};
        Move flip{isPlayerOne, row, col, false,  MoveType::Flip};
        Move hStick{isPlayerOne, row, col, true, MoveType::Stick};
        Move vStick{isPlayerOne, row, col, false, MoveType::Stick};

        int s = completeConnection(board, stone);
        int hS = completeConnection(board, hStick);
        int vS = completeConnection(board, vStick);

        for (int j = 0; j <= s; j++) {
            Move branch = stone;
            branch.stoneChoices = j;
            if (isLegal(board, branch)) gen.push_back(branch);
        }

        if (isLegal(board, flip)) gen.push_back(flip);

        for (int j = 0; j <= hS; j++) {
            Move branch = hStick;
            branch.stoneChoices = i;
            if (isLegal(board, branch)) gen.push_back(branch);
        }

        for (int j = 0; j <= vS; j++) {
            Move branch = vStick;
            branch.stoneChoices = j;
            if (isLegal(board, branch)) gen.push_back(branch);
        }
    }

    return gen;
}

uint64_t Mover::perft(const Board& board, int depth, bool isPlayerOne) {
    if (depth == 0) return 1;
    uint64_t size = 0ULL;

    std::vector<Move> gen = genMoves(board, isPlayerOne);
    if (depth == 1) return gen.size();

    for (auto move : gen) {
        Board next = board;
        applyMove(next, move);
        size += perft(next, depth - 1, !isPlayerOne);
    }

    return size;
}

int Mover::completeConnection(const Board&board, const Move& move) {
    int connect = 0;
    if (move.type == MoveType::Stick) {
        int left = stoneIndex(move.row, move.col);
        int right = stoneIndex(move.row, move.col + 1);
        int up = stoneIndex(move.row, move.col);
        int down = stoneIndex(move.row + 1, move.col);

        if (move.orientH) {
            if ((0 <= move.row && move.row <= 11) && (0 <= move.col && move.col <= 10)) {
                if(board.stoneSet.test(left) && board.stoneSet.test(right)) {
                    return 1;
                }
                return 0;
            }
            return -1;
        } else {
            if ((0 <= move.row && move.row <= 10) && (0 <= move.col && move.col <= 11)) {
                if (board.stoneSet.test(up) && board.stoneSet.test(down)) {
                    return 1;
                }
                return 0;
            }
            return -1;
        }
    } else if (move.type == MoveType::Stone) {
        std::vector<StickRef> neighbors = board.findConnections(move.row, move.col);

        for (auto n : neighbors) {
            int row, col;
            if (n.isH) {
                row = n.index / 11;
                col = n.index % 11;
                if (move.col == col) {
                    int idx = stoneIndex(move.row, col + 1);
                    if (board.stoneSet.test(idx)) {
                        connect++;
                    }
                } else if (move.col == col + 1) {
                    int idx = stoneIndex(move.row, col);
                    if (board.stoneSet.test(idx)) {
                        connect++;
                    }
                }
            } else {
                row = n.index % 11;
                col = n.index / 11;
                if (move.row == row) {
                    int idx = stoneIndex(row + 1, move.col);
                    if (board.stoneSet.test(idx)) {
                        connect++;
                    }
                } else if (move.row == row + 1) {
                    int idx = stoneIndex(row, move.col);
                    if (board.stoneSet.test(idx)) {
                        connect++;
                    }
                }
            }
        }
    } else if (move.type == MoveType::Flip) return 0;

    return connect;
}

int Mover::decode(const StickRef& ref, int r, int c) const {
    int row, col;
    int idx = -1;
    if (ref.isH) {
        col = ref.index % 11;
        if (c == col) {
            idx = stoneIndex(r, col + 1);
        } else if (c == col + 1) {
            idx = stoneIndex(r, col);
        }
    } else {
        row = ref.index % 11;
        if (r == row) {
            idx = stoneIndex(row + 1, c);
        } else if (r == row + 1) {
            idx = stoneIndex(row, c);
        }
    }

    return idx;
}

int Mover::floodFill(const Board& board, int r, int c) {
    int start = stoneIndex(r, c);
    int countVisits = 0;
    std::vector<bool> visited;
    visited.assign(144, false);
    std::deque<int> unchecked;
    visited[start] = true;
    countVisits++;
    unchecked.push_back(start);

    while (!(unchecked.empty())) {
        int look = unchecked.front();
        unchecked.pop_front();
        int row = look / 12;
        int col = look % 12;
        std::vector<StickRef> neighbors = board.findConnections(row, col);

        for (auto n : neighbors) {
            int neighborIdx = decode(n, row, col);
            if (board.stoneSet.test(neighborIdx)) {
                if (board.stoneOwner.test(neighborIdx) == board.stoneOwner.test(look)) {
                    if (visited[neighborIdx] == false) {
                        visited[neighborIdx] = true;
                        countVisits++;
                        unchecked.push_back(neighborIdx);
                    }
                }
            }
        }
    }

    return countVisits;
}

std::vector<int> Mover::stoneNeighbors(const Board& board, int row, int col) const {
    std::vector<int> result;
    std::vector<StickRef> ref = board.findConnections(row, col);
    for (auto r : ref) {
        int idx = decode(r, row, col);
            if (board.stoneSet.test(idx)) {
            result.push_back(idx);
        }
    }
    
    return result;
}

int Mover::score(const Board& board, const Move& move) {
    int totalScore = 0;
    switch (move.type) {
        case MoveType::Stone: {
            int connect = completeConnection(board, move);

            if (connect > 0) {
                totalScore = 2;
                totalScore += connect;
                std::vector<int> neighbors = stoneNeighbors(board, move.row, move.col);
                for (int n : neighbors) {
                    int row = n / 12;
                    int col = n % 12;
                    int size = floodFill(board, row, col);
                    if (size >= 4) {
                        totalScore++;
                        break;
                    }
                }
            }
            break;
        }
        case MoveType::Stick: {
            int connect = completeConnection(board, move);
            totalScore = 1;
            totalScore += connect;

            if (connect > 0) {
                int size = floodFill(board, move.row, move.col);

                if (size >= 4) {
                    totalScore++;
                    break;
                }
            }
            break;
        }
        case MoveType::Flip: {
            return 0;
        }
    }

    return totalScore;
}