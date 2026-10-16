#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <sstream>

#include "../board.h"
#include "../move.h"
#include "infer.h"
#include "match_manager.h"
#include "mcts.h"
MatchResults MatchManager::runGame(
    InferenceEngine* modelA,
    InferenceEngine* modelB,
    int gameIndex,
    int iterationsPerMove,
    std::size_t maxPlies,
    int maxGames,
    bool alternateStartingPlayer,
    bool alternateResourceAdvantage,
    bool reverseResourceAdvantage,
    bool policyPriorsAEnabled,
    bool policyPriorsBEnabled,
    double priorScaleA,
    double priorScaleB
    ) {
    if (iterationsPerMove <= 0 || maxPlies == 0) {
        throw std::invalid_argument("Iterations per move and maximum plies must be positive");
    }

    if (maxGames <= 0) {
        throw std::invalid_argument("Max games must be greater than 0");
    }

    Mover mover;
    MCTS modelAMcts(modelA);
    MCTS modelBMcts(modelB);
    modelAMcts.setPolicyPriorsEnabled(policyPriorsAEnabled);
    modelAMcts.priorScale = priorScaleA;
    modelBMcts.setPolicyPriorsEnabled(policyPriorsBEnabled);
    modelBMcts.priorScale = priorScaleB;
    MatchResults results;

    Board board;
    std::mt19937 gen(47220 + gameIndex / 2);
    const bool playerOneStarts =
        !alternateStartingPlayer || (gameIndex / 2) % 2 == 0;
    board.isPlayerOne = playerOneStarts;
    if (reverseResourceAdvantage) {
        board.sticksOne = 4;
        board.sticksTwo = 3;
    } else if (alternateResourceAdvantage) {
        const bool playerOneHasExtraStick = (gameIndex / 4) % 2 != 0;
        board.sticksOne = playerOneHasExtraStick ? 4 : 3;
        board.sticksTwo = playerOneHasExtraStick ? 3 : 4;
    }
    const bool resourcesDiffer = board.sticksOne != board.sticksTwo;
    const bool playerOneHasExtraStick = board.sticksOne > board.sticksTwo;
    const bool modelAIsPlayerOne = gameIndex % 2 == 0;
    std::size_t plies = 0;
    int winner = 0;

    while (plies < maxPlies) {
        winner = mover.isGameOver(board);
        if (winner != 0) {
            break;
        }

        const bool currentPlayerIsModelA = board.isPlayerOne == modelAIsPlayerOne;
        MCTS& currentMcts = currentPlayerIsModelA ? modelAMcts : modelBMcts;
        Move move;
        if (plies < 8) {
            std::vector<Move> legalMoves = mover.genMoves(board, board.isPlayerOne);

            if (legalMoves.empty()) {
                throw std::runtime_error("No legal moves available during the random opening plies.");
            }

            std::uniform_int_distribution idx(0, static_cast<int>(legalMoves.size()) - 1);
            move = legalMoves[idx(gen)];
        } else {
            move = currentMcts.findBestMove(board, iterationsPerMove);
        }

        if (!mover.isLegal(board, move)) {
            throw std::runtime_error(
                "MCTS returned an illegal move in game " +
                std::to_string(gameIndex + 1));
        }

        mover.applyMove(board, move);
        ++plies;
    }

    if (winner == 0) {
        winner = mover.isGameOver(board);
    }

    const bool endedByScore =
        (board.ptsOne > 30 && board.ptsOne / 2 > board.ptsTwo) ||
        (board.ptsTwo > 30 && board.ptsTwo / 2 > board.ptsOne);

    if (winner == 0) {
        ++results.draws;
        std::cout << "Game " << (gameIndex + 1) << '/' << maxGames
                    << ": draw at " << plies << " plies"
                    << (playerOneStarts
                        ? " (Player 1 started)"
                        : " (Player 2 started)")
                    << (resourcesDiffer
                        ? playerOneHasExtraStick
                            ? " (Player 1 had extra stick)\n"
                            : " (Player 2 had extra stick)\n"
                        : " (equal sticks)\n");
    }

    const bool playerOneWon = winner == 1;
    if (playerOneWon) {
        ++results.playerOneWins;
    } else {
        ++results.playerTwoWins;
    }
    if (playerOneWon == playerOneStarts) {
        ++results.startingPlayerWins;
    } else {
        ++results.nonStartingPlayerWins;
    }
    if (resourcesDiffer) {
        if (playerOneWon == playerOneHasExtraStick) {
            ++results.extraStickHolderWins;
        } else {
            ++results.otherResourcePlayerWins;
        }
        if (playerOneHasExtraStick) {
            if (playerOneWon) {
                ++results.playerOneWinsWithExtraStick;
            } else {
                ++results.playerTwoWinsWithoutExtraStick;
            }
        } else if (playerOneWon) {
            ++results.playerOneWinsWithoutExtraStick;
        } else {
            ++results.playerTwoWinsWithExtraStick;
        }
    }

    const bool modelAWon = playerOneWon == modelAIsPlayerOne;
    if (modelAWon) {
        ++results.modelAWins;
        if (modelAIsPlayerOne) {
            ++results.modelAPlayerOneWins;
        } else {
            ++results.modelAPlayerTwoWins;
        }
    } else {
        ++results.modelBWins;
    }

    std::ostringstream oss;
    oss << "Game " << (gameIndex + 1) << '/' << maxGames
                << ": " << (modelAWon
                    ? modelA != nullptr ? "Model A" : "Heuristic A"
                    : modelB != nullptr ? "Model B" : "Heuristic")
                << " won as Player " << winner
                << (playerOneStarts ? " (Player 1 started)" : " (Player 2 started)")
                << (resourcesDiffer
                    ? playerOneHasExtraStick
                        ? " (Player 1 had extra stick)"
                        : " (Player 2 had extra stick)"
                    : " (equal sticks)")
                << " after " << plies << " plies" << " | Final: points P1=" << board.ptsOne
                << " P2=" << board.ptsTwo
                << ", resources P1=" << board.stonesOne << " stones/"
                << board.sticksOne << " sticks, P2=" << board.stonesTwo
                << " stones/" << board.sticksTwo << " sticks, ended by "
                << (winner == 0 ? "ply limit" : endedByScore ? "score threshold" : "no legal moves");
    {
        std::lock_guard<std::mutex> lk(coutMutex);
        std::cout << oss.str() << std::endl;
    }

    results.gamesPlayed++;

    return results;
}


MatchResults MatchManager::run(
    InferenceEngine* modelA,
    InferenceEngine* modelB,
    std::size_t games,
    int iterationsPerMove,
    std::size_t maxPlies,
    bool alternateStartingPlayer,
    bool alternateResourceAdvantage,
    bool reverseResourceAdvantage,
    bool policyPriorsAEnabled,
    bool policyPriorsBEnabled,
    double priorScaleA,
    double priorScaleB
    ) {
    if (games == 0 || iterationsPerMove <= 0 || maxPlies == 0) {
        throw std::invalid_argument("Games, iterations per move, and maximum plies must be positive");
    }

    MatchResults results;

    for (std::size_t gameIndex = 0; gameIndex < games; ++gameIndex) {
        MatchResults gameResult = runGame(modelA,
            modelB,
            gameIndex,
            iterationsPerMove,
            maxPlies,
            games,
            alternateStartingPlayer,
            alternateResourceAdvantage,
            reverseResourceAdvantage,
            policyPriorsAEnabled,
            policyPriorsBEnabled,
            priorScaleA,
            priorScaleB);
        
        results.modelAWins += gameResult.modelAWins;
        results.modelBWins += gameResult.modelBWins;
        results.draws += gameResult.draws;
        results.modelAPlayerOneWins += gameResult.modelAPlayerOneWins;
        results.modelAPlayerTwoWins += gameResult.modelAPlayerTwoWins;
        results.startingPlayerWins+= gameResult.startingPlayerWins;
        results.playerOneWins += gameResult.playerOneWins;
        results.playerTwoWins += gameResult.playerTwoWins;
        results.nonStartingPlayerWins += gameResult.nonStartingPlayerWins;
        results.extraStickHolderWins += gameResult.extraStickHolderWins;
        results.otherResourcePlayerWins += gameResult.otherResourcePlayerWins;
        results.playerOneWinsWithExtraStick += gameResult.playerOneWinsWithExtraStick;
        results.playerTwoWinsWithExtraStick += gameResult.playerTwoWinsWithExtraStick;
        results.playerOneWinsWithoutExtraStick += gameResult.playerOneWinsWithoutExtraStick;
        results.playerTwoWinsWithoutExtraStick += gameResult.playerTwoWinsWithoutExtraStick;
    }

    return results;
}

MatchResults MatchManager::runSPRT(
    InferenceEngine* modelA,
    InferenceEngine* modelB,
    std::size_t maxGames,
    int iterationsPerMove,
    std::size_t maxPlies,
    bool policyPriorsAEnabled,
    bool policyPriorsBEnabled,
    double priorScaleA,
    double priorScaleB
    ) {
    if (maxGames == 0 || iterationsPerMove <= 0 || maxPlies == 0) {
        throw std::invalid_argument("Max games, iterations per move, and maximum plies must be positive");
    }

    std::cout << "Keep candidate as model A." << std::endl;
    std::cout << "Warning: SPRT does not allow modifications for starting resources." << std::endl;
    std::cout << "Starting SPRT..." << std::endl;
    MatchResults results;
    std::size_t pair = 0;
    double upper = std::log((1 - beta) / alpha);
    double lower = std::log(beta / (1 - alpha));
    double winStep = std::log(p1 / p0);
    double lossStep = std::log((1 - p1) / (1 - p0));

    for (std::size_t gameIndex = 0; gameIndex < maxGames; ++gameIndex) {
        MatchResults gameResult = runGame(
            modelA,
            modelB,
            gameIndex,
            iterationsPerMove,
            maxPlies,
            maxGames,
            true,
            false,
            false,
            policyPriorsAEnabled,
            policyPriorsBEnabled,
            priorScaleA,
            priorScaleB);
        
        results.modelAWins += gameResult.modelAWins;
        results.modelBWins += gameResult.modelBWins;
        results.draws += gameResult.draws;
        results.modelAPlayerOneWins += gameResult.modelAPlayerOneWins;
        results.modelAPlayerTwoWins += gameResult.modelAPlayerTwoWins;
        results.startingPlayerWins+= gameResult.startingPlayerWins;
        results.playerOneWins += gameResult.playerOneWins;
        results.playerTwoWins += gameResult.playerTwoWins;
        results.nonStartingPlayerWins += gameResult.nonStartingPlayerWins;
        results.extraStickHolderWins += gameResult.extraStickHolderWins;
        results.otherResourcePlayerWins += gameResult.otherResourcePlayerWins;
        results.playerOneWinsWithExtraStick += gameResult.playerOneWinsWithExtraStick;
        results.playerTwoWinsWithExtraStick += gameResult.playerTwoWinsWithExtraStick;
        results.playerOneWinsWithoutExtraStick += gameResult.playerOneWinsWithoutExtraStick;
        results.playerTwoWinsWithoutExtraStick += gameResult.playerTwoWinsWithoutExtraStick;
        
        if (gameIndex / 2 == pair) {
            if (gameResult.modelAWins == 1) {
                results.llr += winStep;
            } else if (gameResult.modelBWins == 1) {
                results.llr += lossStep;
            }
        } else {
            pair++;
        }
        if (gameIndex % 2 == 1) {
            if (results.llr >= upper) {
                std::cout << "PASS: Promote this candidate." << std::endl;
                results.promotion = 1;
                results.gamesPlayed = gameIndex + 1;
                break;
            } else if (results.llr <= lower) {
                std::cout << "FAIL: Reject this candidate." << std::endl;
                results.promotion = -1;
                results.gamesPlayed = gameIndex + 1;
                break;
            }
        }

        if ((gameIndex + 1) % 10 == 0) {
            std::cout << "--------------------\n";
            std::cout << "LLR: " << results.llr << std::endl;
            std::cout << "--------------------\n";
        }
    }

    if (results.promotion == 0) {
        std::cout << "INCONCLUSIVE (final LLR " << results.llr << ")" << std::endl;
        results.gamesPlayed = maxGames;
    }


    return results;
}

MatchResults MatchManager::runSPRTParallel(
    InferenceEngine* modelA,
    InferenceEngine* modelB,
    std::size_t maxGames,
    int iterationsPerMove,
    std::size_t maxPlies,
    bool policyPriorsAEnabled,
    bool policyPriorsBEnabled,
    double priorScaleA,
    double priorScaleB
    ) {
    if (maxGames == 0 || iterationsPerMove <= 0 || maxPlies == 0) {
        throw std::invalid_argument("Max games, iterations per move, and maximum plies must be positive");
    }

    std::cout << "Keep candidate as model A." << std::endl;
    std::cout << "Warning: SPRT does not allow modifications for starting resources." << std::endl;
    std::cout << "Starting SPRT..." << std::endl;
    MatchResults results;
    double upper = std::log((1 - beta) / alpha);
    double lower = std::log(beta / (1 - alpha));
    double winStep = std::log(p1 / p0);
    double lossStep = std::log((1 - p1) / (1 - p0));

    std::atomic<std::size_t> nextGame{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::vector<MatchResults> slots(maxGames);
    std::vector<bool> done(maxGames, false);
    std::size_t nextToProcess = 0;
    std::mutex m;
    auto worker = [&] () {
        try {
            while (true) {
                std::size_t idx = nextGame.fetch_add(1);
                if (idx >= maxGames || stop) return;

                MatchResults gameResult = runGame(
                    modelA,
                    modelB,
                    idx,
                    iterationsPerMove,
                    maxPlies,
                    maxGames,
                    true,
                    false,
                    false,
                    policyPriorsAEnabled,
                    policyPriorsBEnabled,
                    priorScaleA,
                    priorScaleB);

                {   // lock scope
                    std::lock_guard<std::mutex> lock(m);
                    slots[idx] = gameResult;
                    done[idx] = true;

                    while (!stop && nextToProcess < maxGames && done[nextToProcess]) {
                        MatchResults g = slots[nextToProcess];
                        results.modelAWins += g.modelAWins;
                        results.modelBWins += g.modelBWins;
                        results.draws += g.draws;
                        results.modelAPlayerOneWins += g.modelAPlayerOneWins;
                        results.modelAPlayerTwoWins += g.modelAPlayerTwoWins;
                        results.startingPlayerWins+= g.startingPlayerWins;
                        results.playerOneWins += g.playerOneWins;
                        results.playerTwoWins += g.playerTwoWins;
                        results.nonStartingPlayerWins += g.nonStartingPlayerWins;
                        results.extraStickHolderWins += g.extraStickHolderWins;
                        results.otherResourcePlayerWins += g.otherResourcePlayerWins;
                        results.playerOneWinsWithExtraStick += g.playerOneWinsWithExtraStick;
                        results.playerTwoWinsWithExtraStick += g.playerTwoWinsWithExtraStick;
                        results.playerOneWinsWithoutExtraStick += g.playerOneWinsWithoutExtraStick;
                        results.playerTwoWinsWithoutExtraStick += g.playerTwoWinsWithoutExtraStick;
                        
                        if (g.modelAWins == 1) {
                            results.llr += winStep;
                        } else if (g.modelBWins == 1) {
                            results.llr += lossStep;
                        }

                        ++nextToProcess;

                        if (results.llr >= upper && nextToProcess % 2 == 0) {
                            std::cout << "PASS: Promote this candidate." << std::endl;
                            results.promotion = 1;
                            results.gamesPlayed = nextToProcess;
                            stop = true;
                            break;
                        } else if (results.llr <= lower && nextToProcess % 2 == 0) {
                            std::cout << "FAIL: Reject this candidate." << std::endl;
                            results.promotion = -1;
                            results.gamesPlayed = nextToProcess;
                            stop = true;
                            break;
                        }

                        if (nextToProcess % 10 == 0) {
                            std::ostringstream llrOss;
                            llrOss << "--------------------"
                            << "LLR: " << results.llr
                            << "--------------------";
                            {
                                std::lock_guard<std::mutex> lk(coutMutex);
                                std::cout << llrOss.str() << std::endl;
                            }
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Worker thread terminated: " << e.what() << std::endl;
            failed = true;
            stop = true;
        } catch (...) {
            std::cerr << "Worker thread terminated." << std::endl;
            failed = true;
            stop = true;
        }
    };

    std::vector<std::thread> parallel;

    for (int i = 0; i < 6; i++) {
        parallel.emplace_back(worker);
    }

    for (auto& t : parallel) {
        t.join();
    }
    if (failed) {
        std::cout << "ERROR: run aborted, result invalid" << std::endl;
    }

    if (results.promotion == 0) {
        std::cout << "INCONCLUSIVE (final LLR " << results.llr << ")" << std::endl;
        results.gamesPlayed = nextToProcess;
    }

    std::cout << "Run completed." << std::endl;

    return results;
}