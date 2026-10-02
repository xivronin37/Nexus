
#include <cmath>
#include <algorithm>

#include "../board.h"
#include "../move.h"
#include "mcts.h"

double MCTS::evaluate(Board& board) {
    int s1 = 0, s2 = 0;
    int S1 = 0, S2 = 0;
    int flippable = 0;
    for (int i = 0; i < 144; i++) {
        if (board.stoneSet.test(i)) {
            int row = i / 12;
            int col = i % 12;
            if (board.stoneOwner.test(i)) {
                s1++;
                if (board.degree(row, col) == 3) {
                    flippable++;
                }
            } else {
                s2++;
                if (board.degree(row, col) == 3) {
                    flippable++;
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
        } else if (board.vStickSet.test(i)) {
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
    double mercyUrgency = 0;

    if (board.ptsOne > 30 && mover.isGameOver(board) == 0) {
        mercyGap = board.ptsTwo - board.ptsOne / 2;
        mercyUrgency = 1.0 / (mercyGap + 0.1);
    } else if (board.ptsTwo > 30 && mover.isGameOver(board) == 0) {
        mercyGap = board.ptsOne - board.ptsTwo / 2;
        mercyUrgency = 1.0 / (mercyGap + 0.1);
    }

    double volatilityFactor = (s1 + s2) > 0 ? (0.6 + sqrt(flippable / (s1 + s2))) + mercyUrgency : 0.8;
    double risk = std::clamp((s1 + s2 + S1 + S2) / 200.0, 0.8, 1.2);

    double rawEval = ((s1 - s2) + (S1 - S2)) * std::pow(volatilityFactor, risk);



    return rawEval;
}

void MCTS::backpropagate(MCTSNode* node, double result) {
    MCTSNode* current = node;
    while (current != nullptr) {
        current->visits++;
        bool nodePlayer = current->board.isPlayerOne;
        current->score += nodePlayer ? result : (1.0 - result);
        current = current->parent;
    }
}

void MCTS::search(MCTSNode* node, Board board) {
    int gameStatus = mover.isGameOver(board);

    if (gameStatus > 0) { // terminal state
        double score = (gameStatus == 1) ? 1.0 : 0.0;
        backpropagate(node, score);
        return;
    }

    double bestUCB1 = -INFINITY;
    MCTSNode* bestChild = nullptr;
    std::vector<Move> legalMoves = mover.genMoves(board, node->board.isPlayerOne);

    if (node->children.size() == legalMoves.size()) { // selection
        for (const auto& child : node->children) {
            if (child->visits == 0) { // prioritize unvisited nodes
                bestChild = child;
                break;
            }

            double UCB1 = (child->score / child->visits) + 1.414 * sqrt(log(node->visits) / child->visits);
            if (UCB1 > bestUCB1) {
                bestUCB1 = UCB1;
                bestChild = child;
            }
        }

        mover.applyMove(board, bestChild->move);
        search(bestChild, board);
    } else { // expansion
        Move unseenMove;
        for (const auto& move : legalMoves) {
            bool exists = false;
            for (const auto& child : node->children) {
                if (move == child->move) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                unseenMove = move;
                break;
            }
        }

        mover.applyMove(board, unseenMove);
        MCTSNode* newChild = new MCTSNode{board, unseenMove, node, {}, 0, 0};
        node->children.push_back(newChild);

        double rawEval = evaluate(board); // rollout

        double evalScore = (std::tanh(rawEval / 400.0) + 1.0) / 2.0;

        backpropagate(newChild, evalScore);

    }
}

void MCTS::deleteTree(MCTSNode* node) {
    for (auto& child : node->children) {
        deleteTree(child);
    }
    delete node;
}

Move MCTS::findBestMove(Board board, int iterations) {
    MCTSNode* root = new MCTSNode{board, Move{}, nullptr, {}, 0, 0};

    for (int i = 0; i < iterations; ++i) {
        search(root, board);
    }

    Move bestMove = root->children.empty() ? Move{} : root->children[0]->move;
    double bestVisits = root->children.empty() ? 0 : root->children[0]->visits;

    for (const auto& child : root->children) {
        if (child->visits > bestVisits) {
            bestVisits = child->visits;
            bestMove = child->move;
        }
    }

    deleteTree(root);

    return bestMove;
}