#pragma once

#include "board.h"
#include "move.h"

class NCP {
    private:
        Board start;
        Mover mover;
    public:
        void runNCP();
};