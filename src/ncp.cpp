
#include <iostream>
#include <sstream>

#include "ncp.h"

void NCP::runNCP() {
     while (true) {
        std::string line;
        std::getline(std::cin, line);

        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "ncp") {
            std::cout << "name Nexus 1.0" << "\n" << std::flush;
            std::cout << "author MC" << "\n" << std::flush;
            std::cout << "ncpok" << "\n" << std::flush;
        } else if (command == "go") {
            // engine not implemented yet
        } else if (command == "halt") {
            // engine not implemented yet
        } else if (command == "stop") {
            break;
        } else if (command == "pos") {
            std::string token;
            if (ss >> token && token == "play") {
                /* while (ss >> token) {
                    
                } */
            }
        } else if (command == "move") {
            std::string token, one;
            ss >> token;
            if (token == "stone") {
                ss >> one;
                std::pair<int, int> coord = start.parseCoord(one);
                int bonus = 0;
                bool catchError = false;
                if (ss >> token) {
                    try {
                        for (char c : token) {
                            if (!std::isdigit(c)) {
                                throw std::runtime_error("");
                            }
                        }

                        bonus = std::stoi(token);
                    } catch (...) {
                        catchError = true;
                        std::cout << "Error occured while parsing bonus\n" << std::endl;
                    }
                }
                if (!catchError) {
                    Move stone{start.isPlayerOne, coord.first, coord.second, false, MoveType::Stone, bonus};
                    if (mover.isLegal(start, stone)) {
                        mover.applyMove(start, stone);
                        std::cout << "moveok\n" << std::flush;
                    } else {
                        std::cout << "illegalMove\n" << std::flush;
                    }
                }

            } else if (token == "stick") {
                ss >> token;
                int idx = token.find('-');
                std::string first = token.substr(0, idx);
                std::string second = token.substr(idx + 1);
                std::pair<int, int> coordOne = start.parseCoord(first);
                std::pair<int, int> coordTwo = start.parseCoord(second);
                

                int bonus = 0;
                if (ss >> token) {
                    bonus = std::stoi(token);
                }

                bool orientH;

                if (coordOne.first == coordTwo.first) {
                    orientH = true;
                } else if (coordOne.second == coordTwo.second) {
                    orientH = false;
                }

                Move stick{start.isPlayerOne, coordOne.first, coordOne.second, orientH, MoveType::Stick, bonus};
                if (mover.isLegal(start, stick)) {
                    mover.applyMove(start, stick);
                    std::cout << "moveok\n" << std::flush;
                } else {
                    std::cout << "illegalMove\n" << std::flush;
                }

            } else if (token == "flip") {
                ss >> one;
                std::pair<int, int> coord = start.parseCoord(one);
                int bonus = 0;
                if (ss >> token) {
                    bonus = std::stoi(token);
                }
                Move flip{start.isPlayerOne, coord.first, coord.second, false, MoveType::Flip, bonus};
                if (mover.isLegal(start, flip)) {
                    mover.applyMove(start, flip);
                    std::cout << "moveok\n" << std::flush;
                } else {
                    std::cout << "illegalMove\n" << std::flush;
                }
            } 
        }
    }
}