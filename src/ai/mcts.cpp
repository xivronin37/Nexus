
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>

#include "../board.h"
#include "../move.h"
#include "infer.h"
#include "mcts.h"

std::size_t policyActionIndex(const Move& move) {
    if (move.row < 0 || move.row >= 12 || move.col < 0 || move.col >= 12) {
        throw std::out_of_range("Move coordinate is outside the policy action map");
    }

    const std::size_t square =
        static_cast<std::size_t>(move.row * 12 + move.col);
    switch (move.type) {
        case MoveType::Stone:
            return square;
        case MoveType::Flip:
            return 144 + square;
        case MoveType::Stick:
            if (move.orientH) {
                if (move.col >= 11) {
                    throw std::out_of_range(
                        "Horizontal stick is outside the policy action map");
                }
                return 288 + static_cast<std::size_t>(move.row * 11 + move.col);
            }
            if (move.row >= 11) {
                throw std::out_of_range(
                    "Vertical stick is outside the policy action map");
            }
            return 420 + static_cast<std::size_t>(move.col * 11 + move.row);
    }
    throw std::logic_error("Unknown move type in policy action map");
}

double MCTS::evaluate(Board& board) {
    if (evaluator != nullptr) {
        return evaluator->predict(board);
    }

    int s1 = 0, s2 = 0;
    int S1 = 0, S2 = 0;
    int flippable = 0;
    int flippableSelf = 0;
    int flippableOpp = 0;  
    for (int i = 0; i < 144; i++) {
        if (board.stoneSet.test(i)) {
            int row = i / 12;
            int col = i % 12;
            int left = row > 0 ? stoneIndex(row - 1, col) : -1;
            int right = row < 11 ? stoneIndex(row + 1, col) : -1;
            int up = col > 0 ? stoneIndex(row, col - 1) : -1;
            int down = col < 11 ? stoneIndex(row, col + 1) : -1;

            if (board.stoneOwner.test(i)) {
                if (board.degree(row, col) > 0) {
                    s1++;
                }
                if (board.degree(row, col) == 3) {
                    flippable++;
                    flippableSelf++;
                }
                if (left != -1 && board.stoneSet.test(left) && board.stoneOwner.test(left)) {
                    s1++;
                }
                if (right != -1 && board.stoneSet.test(right) && board.stoneOwner.test(right)) {
                    s1++;
                }
                if (up != -1 && board.stoneSet.test(up) && board.stoneOwner.test(up)) {
                    s1++;
                }
                if (down != -1 && board.stoneSet.test(down) && board.stoneOwner.test(down)) {
                    s1++;
                }
            } else {
                if (board.degree(row, col) > 0) {
                    s2++;
                }
                if (board.degree(row, col) == 3) {
                    flippable++;
                    flippableOpp++;
                }
                if (left != -1 && board.stoneSet.test(left) && !board.stoneOwner.test(left)) {
                    s2++;
                }
                if (right != -1 && board.stoneSet.test(right) && !board.stoneOwner.test(right)) {
                    s2++;
                }
                if (up != -1 && board.stoneSet.test(up) && !board.stoneOwner.test(up)) {
                    s2++;
                }
                if (down != -1 && board.stoneSet.test(down) && !board.stoneOwner.test(down)) {
                    s2++;
                }
            }
        }
    }

    for (int i = 0; i < 132; i++) {
        if (board.hStickSet.test(i)) {
            int row = i / 11;
            int col = i % 11;
            if (board.hStickOwner.test(i)) {
                S1++;
            } else {
                S2++;
            }
        }

        if (board.vStickSet.test(i)) {
            int row = i % 11;
            int col = i / 11;
            if (board.vStickOwner.test(i)) {
                S1++;
            } else {
                S2++;
            }
        }
    }

    double mercyGap = 0;

    double mercySignal = 0;
    if (board.ptsOne > 30 && mover.isGameOver(board) == 0) {
        double gap = board.ptsTwo - board.ptsOne / 2.0; // negative = P1 close to winning
        mercySignal = -50.0 / (gap - 0.1); // gap near 0 or negative => large positive, favors P1
    } else if (board.ptsTwo > 30 && mover.isGameOver(board) == 0) {
        double gap = board.ptsOne - board.ptsTwo / 2.0;
        mercySignal = 50.0 / (gap - 0.1); // explodes negative, favors P2
    }

    int poolOne = board.stonesOne + board.sticksOne;
    int poolTwo = board.stonesTwo + board.sticksTwo;

    double scarcitySignal = 0;
    scarcitySignal -= 15.0 / (poolOne + 0.5);  // P1 running low pulls eval down (bad for P1)
    scarcitySignal += 15.0 / (poolTwo + 0.5); // same logic for P2

    double pointsSignal = board.ptsOne - board.ptsTwo;
    double flipSignal = (flippableOpp - flippableSelf) * 0.6;

    double minResources = std::min((s1 + S1), (s2 + S2));
    double ptsGap = minResources > 0 ? std::abs(0 - minResources) : 1000.0;
    const int activeResourceDifference = (s1 + S1) - (s2 + S2);
    double resourceGap = 0.0;
    if (activeResourceDifference > 0) {
        resourceGap = 25.0 / ptsGap;
    } else if (activeResourceDifference < 0) {
        resourceGap = -25.0 / ptsGap;
    }
    double volatilityFactor = (s1 + s2) > 0 ? (0.6 + sqrt(flippable / (s1 + s2))) : 0.8;
    double risk = std::clamp((s1 + s2 + S1 + S2) / 200.0, 0.8, 1.2);

    double rawEval = 0.3 * ((s1 - s2) + (S1 - S2) + resourceGap) * std::pow(volatilityFactor, risk) + mercySignal + scarcitySignal + pointsSignal
    + flipSignal;

    return board.isPlayerOne ? rawEval : -rawEval;
}

void MCTS::backpropagate(MCTSNode* node, double result) {
    MCTSNode* current = node;
    while (current != nullptr) {
        current->visits++;
        bool nodePlayer = current->move.isPlayerOne;
        current->score += nodePlayer ? result : (1.0 - result);
        current = current->parent;
    }
}

void MCTS::search(MCTSNode* node, Board board, int depth) {
    if (depth > maxDepth) {
        maxDepth = depth;
    }

    int gameStatus = mover.isGameOver(board);

    if (gameStatus > 0) { // terminal state
        double score = (gameStatus == 1) ? 1.0 : 0.0;
        backpropagate(node, score);
        return;
    }

    if (!node->slotsBuilt) {
        std::vector<Move> legalMoves = mover.genMovesSearch(node->board, node->board.isPlayerOne);
        std::vector<double> rawScores;

        if (legalMoves.empty()) {
            throw std::runtime_error("Search position has no generated legal moves");
        }

        if (evaluator != nullptr && evaluator->hasPolicyHead() &&
            policyPriorsEnabled) {
            const std::vector<float> policyLogits =
                evaluator->predictPolicy(node->board);
            rawScores.reserve(legalMoves.size());
            for (const Move& move : legalMoves) {
                rawScores.push_back(0); // temporary
            }
        } else if (evaluator != nullptr) {
            std::vector<Board> successorBoards;
            successorBoards.reserve(legalMoves.size());
            for (Move move : legalMoves) {
                Board next = node->board;
                mover.applyMove(next, move);
                successorBoards.push_back(next);
            }

            const std::vector<float> predictions =
                evaluator->predictBatch(successorBoards);
            for (float prediction : predictions) {
                rawScores.push_back(-prediction * priorScale);
            }
        } else {
            for (auto& move : legalMoves) {
                Board next = node->board;
                mover.applyMove(next, move);
                double raw = evaluate(next);
                rawScores.push_back(-raw);
            }
        }

        double maxScore = *std::max_element(rawScores.begin(), rawScores.end());
        double sumExp = 0.0;
        std::vector<double> expScores;
        for (double score : rawScores) {
            double expScore = std::exp(score - maxScore);
            expScores.push_back(expScore);
            sumExp += expScore;
        }
        for (int i = 0; i < legalMoves.size(); i++) {
            double prior = expScores[i] / sumExp;
            node->slots.push_back(MoveSlot{legalMoves[i], prior, nullptr});
        }
    
        node->slotsBuilt = true;
    }


    int bestIdx = -1;
    double bestPUCT = -INFINITY;
    for (int i = 0; i < node->slots.size(); i++) {
        double q = node->slots[i].child ? (node->slots[i].child->score / node->slots[i].child->visits) : 0.0;
        double visits = node->slots[i].child ? node->slots[i].child->visits : 0.0;
        double u = 1.414 * node->slots[i].prior * sqrt(node->visits) / (1.0 + visits);
        double puct = q + u;

        if (puct > bestPUCT) {
            bestPUCT = puct;
            bestIdx = i; 
        }
    }

    MoveSlot& chosen = node->slots[bestIdx];

    if (chosen.child == nullptr) {
        chosen.child = new MCTSNode{board, chosen.move, node, {}, 0, 0};
        Board testBoard = board;
        mover.applyMove(testBoard, chosen.move);
        chosen.child->board = testBoard;
        double rawEval = evaluate(testBoard);
        const double playerOneEval = testBoard.isPlayerOne
            ? rawEval
            : -rawEval;
        double evalScore = evaluator != nullptr
            ? (std::clamp(playerOneEval, -1.0, 1.0) + 1.0) / 2.0
            : (std::tanh(playerOneEval / 400.0) + 1.0) / 2.0;
        backpropagate(chosen.child, evalScore);
    } else {
        mover.applyMove(board, chosen.move);
        search(chosen.child, board, depth + 1);
    }
}

void MCTS::deleteTree(MCTSNode* node) {
    for (auto& slot : node->slots) {
        if (slot.child != nullptr) {
            deleteTree(slot.child);
        }
    }
    delete node;
}

Move MCTS::findBestMove(Board board, int iterations) {
    maxDepth = 0;
    MCTSNode* root = new MCTSNode{board, Move{}, nullptr, {}, 0, 0};

    for (int i = 0; i < iterations; ++i) {
        search(root, board);
    }

    Move bestMove =  root->slots.empty() ? Move{} : root->slots[0].move;
    double bestVisits = -1;
    for (const auto& slot : root->slots) {
        double visits = slot.child ? slot.child->visits : 0;
        double score = slot.child ? slot.child->score : 0;

        if (visits > bestVisits) {
            bestVisits = visits;
            bestMove = slot.move;
        }
    }

    deleteTree(root);

    return bestMove;
}

PolicySearchResult MCTS::findBestMoveWithVisitDistribution(Board board,int iterations, double temperature) {
    if (!std::isfinite(temperature) || temperature < 0.0) {
        throw std::invalid_argument(
            "MCTS temperature must be finite and non-negative");
    }
    if (iterations < 1) {
        throw std::invalid_argument("MCTS iterations must be at least 1");
    }

    MCTSNode* root = new MCTSNode{board, Move{}, nullptr, {}, 0, 0};
    if (temperature > 0.0) {
        search(root, board, 0);
        std::vector<double> noises(root->slots.size());
        std::gamma_distribution<double> gammaDist(0.1, 1.0);
        for (double& noise : noises) {
            noise = gammaDist(gen);
        }
        const double noiseSum =
            std::accumulate(noises.begin(), noises.end(), 0.0);
        constexpr double epsilon = 0.25;
        if (noiseSum > 0.0) {
            for (std::size_t i = 0; i < root->slots.size(); ++i) {
                root->slots[i].prior =
                    (1.0 - epsilon) * root->slots[i].prior
                    + epsilon * noises[i] / noiseSum;
            }
        }
        for (int i = 1; i < iterations; ++i) {
            search(root, board, 0);
        }
    } else {
        for (int i = 0; i < iterations; ++i) {
            search(root, board);
        }
    }

    if (root->slots.empty()) {
        deleteTree(root);
        throw std::runtime_error("MCTS search found no legal root moves");
    }

    double totalVisits = 0.0;
    for (const auto& slot : root->slots) {
        totalVisits += slot.child ? slot.child->visits : 0.0;
    }
    if (totalVisits <= 0.0) {
        deleteTree(root);
        throw std::runtime_error(
            "MCTS search produced no visited root moves");
    }

    std::size_t bestIndex = 0;
    double bestVisits = -1.0;
    double maxLogVisits = 0.0;
    for (const auto& slot : root->slots) {
        const double visits = slot.child ? slot.child->visits : 0.0;
        if (visits > bestVisits) {
            bestVisits = visits;
        }
    }
    if (temperature > 0.0 && bestVisits > 0.0) {
        maxLogVisits = std::log(bestVisits);
    }

    std::vector<double> selectionWeights;
    selectionWeights.reserve(root->slots.size());
    std::map<std::size_t, double> actionVisits;
    for (std::size_t i = 0; i < root->slots.size(); ++i) {
        const auto& slot = root->slots[i];
        const double visits = slot.child ? slot.child->visits : 0.0;
        if (i == 0 || visits > (root->slots[bestIndex].child
                ? root->slots[bestIndex].child->visits : 0.0)) {
            bestIndex = i;
        }
        actionVisits[policyActionIndex(slot.move)] += visits;
        if (temperature > 0.0 && visits > 0.0) {
            selectionWeights.push_back(
                std::exp((std::log(visits) - maxLogVisits) / temperature));
        } else {
            selectionWeights.push_back(visits);
        }
    }

    if (temperature > 0.0) {
        std::discrete_distribution<std::size_t> distribution(
            selectionWeights.begin(), selectionWeights.end());
        bestIndex = distribution(gen);
    }

    PolicySearchResult result{root->slots[bestIndex].move, {}};
    result.visitDistribution.reserve(actionVisits.size());
    for (const auto& [action, visits] : actionVisits) {
        result.visitDistribution.emplace_back(action, visits / totalVisits);
    }
    deleteTree(root);
    return result;
}

Move MCTS::stochasticFindBestMove(Board board, int iterations, double temperature) {
    if (temperature == 0.0) {
        return findBestMove(board, iterations);
    }
    if (!std::isfinite(temperature) || temperature < 0.0) {
        throw std::invalid_argument(
            "MCTS temperature must be finite and non-negative");
    }

    MCTSNode* root = new MCTSNode{board, Move{}, nullptr, {}, 0, 0};

    search(root, board, 0);
    std::vector<double> noises(root->slots.size());
    std::gamma_distribution<double> gammaDist(0.1, 1.0);
    for (size_t i = 0; i < root->slots.size(); ++i) {
        double noise = gammaDist(gen);
        noises[i] = noise;
    }

    double noiseSum = std::accumulate(noises.begin(), noises.end(), 0.0);
    double epsilon = 0.25; // force some noise into priors to discourage always choosing the same move
    
    if (noiseSum > 0.0) {
        for (size_t i = 0; i < root->slots.size(); ++i) {
            root->slots[i].prior = (1 - epsilon) * root->slots[i].prior + (epsilon * noises[i] / noiseSum);
        }
    }

    for (int i = 1; i < iterations; ++i) {
        search(root, board, 0);
    }

    double maxVisits = 0.0;
    for (const auto& slot : root->slots) {
        if (slot.child != nullptr) {
            maxVisits = std::max(maxVisits, slot.child->visits);
        }
    }
    if (maxVisits <= 0.0) {
        deleteTree(root);
        throw std::runtime_error(
            "MCTS search produced no visited root moves");
    }

    const double maxLogVisits = std::log(maxVisits);
    std::vector<double> weights;
    weights.reserve(root->slots.size());
    for (auto& slot : root->slots) {
        double visits = slot.child ? slot.child->visits : 0.0;
        const double weight = visits > 0.0
            ? std::exp((std::log(visits) - maxLogVisits) / temperature)
            : 0.0;
        weights.push_back(weight);
    }

    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int chosenIdx = dist(gen); 

    Move chosen = root->slots[chosenIdx].move;
    deleteTree(root);
    return chosen;
}

Move MCTS::pickMove(Board board, int idx) {
    std::vector<Move> legalMoves = mover.genMoves(board, board.isPlayerOne);
    return legalMoves[idx];
}