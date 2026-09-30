
#include <cctype>
#include <stdexcept>

#include "board.h"

int stoneIndex(int row, int col) {
    return row * 12 + col;
}

int hStickIndex(int row, int col) {
    return row * 11 + col;
}

int vStickIndex(int row, int col) {
    return col * 11 + row;
}

int Board::degree(int row, int col) const {
    int degree = 0;
    int left = hStickIndex(row, col - 1);
    int right = hStickIndex(row, col);
    int up = vStickIndex(row - 1, col);
    int down = vStickIndex(row, col);

    if (col != 0 && hStickSet.test(left)) {
        degree++;
    }
    if (col != 11 && hStickSet.test(right)) {
        degree++;
    }
    if (row != 0 && vStickSet.test(up)) {
        degree++;
    }
    if (row != 11 && vStickSet.test(down)) {
        degree++;
    }

    return degree;
}

std::vector<StickRef> Board::findConnections(int row, int col) const {
    std::vector<StickRef> connections;
    int left = hStickIndex(row, col - 1);
    int right = hStickIndex(row, col);
    int up = vStickIndex(row - 1, col);
    int down = vStickIndex(row, col);

    if (col != 0 && hStickSet.test(left)) {
        connections.push_back(StickRef{true, left});
    }
    if (col != 11 && hStickSet.test(right)) {
        connections.push_back(StickRef{true, right});
    }
    if (row != 0 && vStickSet.test(up)) {
        connections.push_back(StickRef{false, up});
    }
    if (row != 11 && vStickSet.test(down)) {
        connections.push_back(StickRef{false, down});
    }

    return connections;
}

std::pair<int, int> Board::parseCoord(const std::string& s) {
    if (s.size() <= 1) return {-1, -1};
    int rank = 0;
    try {
        std::string rankStr = s.substr(1);
        for (char c : rankStr) {
            if (!std::isdigit(c)) {
                throw std::runtime_error("Invalid move");
            }
        }
        rank = std::stoi(rankStr);
    } catch (...){
        return {-1, -1};
    }

    char fileChar = s[0];

    if (fileChar < 'a' || fileChar > 'l' || rank < 1 || rank > 12) {
        return {-1, -1}; 
    }

    int file = fileChar - 'a';
    rank--;
    return {file, rank};
}