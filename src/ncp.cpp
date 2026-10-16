
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <map>

#include "ncp.h"
#include "ai/infer.h"

void NCP::runNCP() {
    while (true) {
        std::string line;
        std::getline(std::cin, line);

        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "ncp") {
            std::cout << "name Arcturus 2.0" << "\n" << std::flush;
            std::cout << "author MC" << "\n" << std::flush;
            std::cout << "ncpok" << "\n" << std::flush;
        } else if (command == "go") {
            int iterations = 1000; // default
            double temperature = 1.0; // default
            bool isStochastic = false;
            bool includeVisitDistribution = false;
            std::string token;

            if (ss >> token) {
                try {
                    iterations = std::stoi(token);
                } catch (...) {
                    // malformed, just keep default
                }
            }
            if (ss >> token) {
                try {
                    temperature = std::stod(token);
                    isStochastic = true;
                } catch (...) {
                    includeVisitDistribution = token == "visits";
                }
            }
            if (ss >> token) {
                includeVisitDistribution = token == "visits";
            }
            
            Move move;
            PolicySearchResult searchResult;

            if (includeVisitDistribution) {
                searchResult = ai.findBestMoveWithVisitDistribution(
                    start, iterations, isStochastic ? temperature : 1.0);
                move = searchResult.move;
            } else if (isStochastic) {
                move = ai.stochasticFindBestMove(start, iterations, temperature);
            } else {
                move = ai.findBestMove(start, iterations);
            }

            mover.applyMove(start, move);
            std::cout << "moveok\n" << std::flush;
            std::string bestMove = "Move";
            if (move.type == MoveType::Stone) {
                bestMove += " stone ";
                bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
            } else if (move.type == MoveType::Stick) {
                bestMove += " stick ";
                bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
                bestMove += " ";
                if (move.orientH) {
                    bestMove += std::string(1, 'a' + move.col + 1) + std::to_string(move.row + 1);
                } else {
                    bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 2);
                }
            } else if (move.type == MoveType::Flip) {
                bestMove += " flip ";
                bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
            }

            bestMove += " " + std::to_string(move.stoneChoices);
            std::cout << bestMove << "\n" << std::flush;
            if (includeVisitDistribution) {
                std::cout << "visits" << std::setprecision(17);
                for (const auto& [action, probability] : searchResult.visitDistribution) {
                    std::cout << " " << action << ":" << probability;
                }
                std::cout << "\n" << std::flush;
            }

        } else if (command == "halt") {
            // engine not implemented yet
        } else if (command == "stop") {
            break;
        } else if (command == "pos") {
            start.reset();
            std::cout << "posok" << std::endl;
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
                std::pair<int, int> coordOne = start.parseCoord(token);
                ss >> token;
                std::pair<int, int> coordTwo = start.parseCoord(token);
                

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
        } else if (command == "export") {
            std::string stones = start.stoneSet.to_string();
            std::string stoneOwner = start.stoneOwner.to_string();
            std::string hSticks = start.hStickSet.to_string();
            std::string hStickOwner = start.hStickOwner.to_string();
            std::string vSticks = start.vStickSet.to_string();
            std::string vStickOwner = start.vStickOwner.to_string();
            std::cout << stones << ' '
                      << stoneOwner << ' '
                      << hSticks << ' '
                      << hStickOwner << ' '
                      << vSticks << ' '
                      << vStickOwner << ' '
                      << (start.isPlayerOne ? 1 : 0) << ' '
                      << start.stonesOne << ' '
                      << start.sticksOne << ' '
                      << start.ptsOne << ' '
                      << start.stonesTwo << ' '
                      << start.sticksTwo << ' '
                      << start.ptsTwo;
            for (std::size_t i = 0; i < 6; ++i) {
                const int flippedIndex =
                    i < start.recentlyFlipped.size()
                        ? start.recentlyFlipped[i]
                        : -1;
                std::cout << ' ' << flippedIndex;
            }
            std::cout << '\n' << std::flush;
            // warning: reversed bit mapping, so 143 would map to 0 index here.
        } else if (command == "status") {
            std::cout << mover.isGameOver(start) << std::endl;
        } else if (command == "eval") {
            Board copy = start;
            if (mover.isGameOver(copy) != 0) {
                std::cout << "null" << std::endl;
            } else {
                double score = ai.evaluate(copy);
                std::cout << score << std::endl;
            }
        } else if (command == "priors") {
            Board copy = start;
            std::vector<Move> legalMoves = mover.genMoves(copy, copy.isPlayerOne);
            if (legalMoves.empty()) {
                std::cout << "empty" << std::endl;
            } else {
                std::vector<Board> successorBoards;

                for (Move m : legalMoves) {
                    Board nextBoard = copy;
                    mover.applyMove(nextBoard, m);
                    successorBoards.push_back(nextBoard);
                }
                std::vector<float> scores = ai.evaluator->predictBatch(successorBoards);
                std::map<std::size_t, double> scoresMap;

                for (int i = 0; i < legalMoves.size(); i++) {
                    std::size_t idx = policyActionIndex(legalMoves[i]);
                    double current = -scores[i];
                    auto it = scoresMap.find(idx);

                    if (it == scoresMap.end()) {
                        scoresMap[idx] = current;
                    } else {
                        if (it->second <= current) {
                            it->second = current;
                        }
                    }
                }

                std::cout << "scores " << std::setprecision(17);

                for (const auto& [key, value] : scoresMap) {
                    std::cout << key << ":" << value << " ";
                }
                std::cout << std::endl;
            }
        } else if (command == "random") {
            std::string token;

            if (ss >> token) {
                std::vector<Move> legalMoves = mover.genMoves(start, start.isPlayerOne);
                try {
                    size_t idx = std::stoi(token) % legalMoves.size();
                    Move move = legalMoves[idx];

                    if (legalMoves.empty()) {
                        std::cout << "No legal moves available in this state.\n" << std::flush;
                        continue; 
                    }

                    mover.applyMove(start, move);
                    std::cout << "moveok\n" << std::flush;
                    std::string bestMove = "Move";
                    if (move.type == MoveType::Stone) {
                        bestMove += " stone ";
                        bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
                    } else if (move.type == MoveType::Stick) {
                        bestMove += " stick ";
                        bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
                        bestMove += " ";
                        if (move.orientH) {
                            bestMove += std::string(1, 'a' + move.col + 1) + std::to_string(move.row + 1);
                        } else {
                            bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 2);
                        }
                    } else if (move.type == MoveType::Flip) {
                        bestMove += " flip ";
                        bestMove += std::string(1, 'a' + move.col) + std::to_string(move.row + 1);
                    }

                    bestMove += " " + std::to_string(move.stoneChoices);
                    std::cout << bestMove << "\n" << std::flush;
                } catch (...) {
                    std::cout << "Error in string-to-integer conversion." << std::endl;
                }

            } else {
                std::cout << "No seed provided." << std::endl;
            }
        }
    }
}