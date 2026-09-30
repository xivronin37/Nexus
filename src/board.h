#pragma once

#include <cstdint>
#include <bitset>
#include <vector>

int stoneIndex(int row, int col);
int hStickIndex(int row, int col);
int vStickIndex(int row, int col);

struct StickRef {
    bool isH;
    int index;
};

class Board {
    public:
        bool isPlayerOne = true;
        std::bitset<144> stoneSet{}, stoneOwner{};
        std::bitset<132> hStickSet{}, vStickSet{}, hStickOwner{}, vStickOwner{};
        int stonesOne = 5;
        int sticksOne = 3;
        int stonesTwo = 5;
        int sticksTwo = 4;
        int ptsOne = 0;
        int ptsTwo = 0;
        int degree(int row, int col) const;
        std::vector<StickRef> findConnections(int row, int col) const;
        std::pair<int, int> parseCoord(const std::string& s);
};
