#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include "board.h"
#include "move.h"
#include "ai/infer.h"
#include "ai/mcts.h"
    
std::random_device rd;
std::mt19937 gen(rd());

ma_engine engine;
ma_result result = ma_engine_init(NULL, &engine);

struct GameState {
    Board board;
    bool us = true;
    bool aiMoved = false;
    bool awaitingBonus = false;
    int bonusSliderValue = 0;
    Move pendingMove;
    std::string bonusError;
    bool stoneMode = true;
    bool stickMode = false;
    int winner = 0;
    double thinkTime = 0.0;
};

int main(int argc, char* argv[]) {
    if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--model")) {
        std::cerr << "Usage: NexusGui [--model <model.onnx>]\n";
        return 2;
    }

    std::unique_ptr<InferenceRuntime> runtime;
    std::unique_ptr<InferenceEngine> evaluator;
    if (argc == 3) {
        try {
            runtime = std::make_unique<InferenceRuntime>();
            evaluator = std::make_unique<InferenceEngine>();
            if (!evaluator->load(argv[2], *runtime)) {
                return 1;
            }
        } catch (const std::exception& error) {
            std::cerr << "Neural model startup failed: "
                      << error.what() << '\n';
            return 1;
        }
    }

    if (result != MA_SUCCESS) {
        std::cout << "Failed to initialize audio engine." << std::endl;
    }

    ma_engine_set_volume(&engine, 0.6f);
    
    Mover mover;
    MCTS ai;
    if (evaluator != nullptr) {
        ai.setEvaluator(evaluator.get());
    }
    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "Failed to initialize GLFW.\n";
        return 1;
    }
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Nexus", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create the Nexus window.\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    ImGui::StyleColorsDark();  // start from the built-in dark theme as a base
    ImGuiStyle& style = ImGui::GetStyle();

    float cellSize = 45.0f;
    int gridSize = 12;

    bool stoneMode = true;
    bool stickMode = false;
    static bool endSoundPlayed = false;

    GameState game;

    while (!glfwWindowShouldClose(window)) {
        bool moveApplied = false;

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImVec4 grayBg = ImVec4(128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, grayBg);
        
        ImGui::Begin("Board");

        ImVec2 origin = ImGui::GetCursorScreenPos();
        origin.x += 5.0f;
        origin.y += 10.0f;
        ImVec2 boardTopLeft = origin;
        ImU32 lineColor = IM_COL32(0, 0, 0, 255);
        ImU32 boardColor = IM_COL32(222, 184, 135, 255);

        ImVec2 boardBottomRight(origin.x + (gridSize - 1) * cellSize, origin.y + (gridSize - 1) * cellSize);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(boardTopLeft, boardBottomRight, boardColor);
        for (int col = 0; col < gridSize; col++) {
            float x = origin.x + col * cellSize;
            ImVec2 top(x, origin.y);
            ImVec2 bottom(x, origin.y + (gridSize - 1) * cellSize);
            drawList->AddLine(top, bottom, lineColor);
        }

        for (int row = 0; row < gridSize; row++) {
            float y = origin.y + row * cellSize;
            ImVec2 left(origin.x, y);
            ImVec2 right(origin.x + (gridSize - 1) * cellSize, y);
            drawList->AddLine(left, right, lineColor);
        }

        for (int i = 0; i < 132; i++) {
            if (game.board.hStickSet.test(i)) {
                int r = i / 11;
                int c = i % 11;

                float startX = origin.x + (c * cellSize);
                float startY = origin.y + (r * cellSize);
                
                ImVec2 startPos(startX, startY);
                ImVec2 endPos(startX + cellSize, startY);
                
                ImU32 stickColor = game.board.hStickOwner.test(i) ? IM_COL32(255, 255, 255, 255) : IM_COL32(0, 0, 0, 255); 
                drawList->AddLine(startPos, endPos, stickColor, 3.5f);
            }
        }

        for (int i = 0; i < 132; i++) {
            if (game.board.vStickSet.test(i)) {
                int r = i % 11;
                int c = i / 11;
                
                float startX = origin.x + (c * cellSize);
                float startY = origin.y + (r * cellSize);
                
                ImVec2 startPos(startX, startY);
                ImVec2 endPos(startX, startY + cellSize);
                
                ImU32 stickColor = game.board.vStickOwner.test(i) ? IM_COL32(255, 255, 255, 255) : IM_COL32(0, 0, 0, 255); ; 
                drawList->AddLine(startPos, endPos, stickColor, 3.5f);
            }
        }

        for (int i = 0; i < 144; i++) {
            int row = i / gridSize;
            int col = i % gridSize;

            float intersectX = origin.x + col * cellSize;
            float intersectY = origin.y + row * cellSize;

            ImVec2 intersectPos(intersectX, intersectY);
            /* if (i == 1) {  // hardcode the specific index you're testing, temporarily
                std::cout << "Draw loop sees owner=" << game.board.stoneOwner.test(i) << "\n";
            } */

            if (game.board.stoneSet.test(i)) {
                ImU32 stoneColor = game.board.stoneOwner.test(i) ? IM_COL32(255, 255, 255, 255) : IM_COL32(0, 0, 0, 255);
                drawList->AddCircleFilled(intersectPos, cellSize * 0.30f, stoneColor);
            }
        }

        ImVec2 mousePos = ImGui::GetMousePos();

        float relativeX = mousePos.x - origin.x;
        float relativeY = mousePos.y - origin.y;
        int hoveredCol = static_cast<int>(std::round(relativeX / cellSize));
        int hoveredRow = static_cast<int>(std::round(relativeY / cellSize));
        float offsetX = relativeX - (hoveredCol * cellSize);
        float offsetY = relativeY - (hoveredRow * cellSize);
        if (game.winner == 0 && !game.awaitingBonus &&
            hoveredRow >= 0 && hoveredRow < gridSize &&
            hoveredCol >= 0 && hoveredCol < gridSize) { // flip
    
            int idx = stoneIndex(hoveredRow, hoveredCol);

            if (game.board.stoneSet.test(idx) && game.board.stoneOwner.test(idx) != game.board.isPlayerOne) {
                if (game.board.degree(hoveredRow, hoveredCol) == 3) {
                    float snapX = origin.x + (hoveredCol * cellSize);
                    float snapY = origin.y + (hoveredRow * cellSize);
                    ImVec2 snapPos(snapX, snapY);

                    ImU32 previewColor = game.board.isPlayerOne ? IM_COL32(255, 255, 255, 120) : IM_COL32(0, 0, 0, 120); 
                    drawList->AddCircleFilled(snapPos, cellSize * 0.43f, previewColor);

                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !moveApplied) {
                        Move move{game.board.isPlayerOne, hoveredRow, hoveredCol, false, MoveType::Flip};
                        if (mover.isLegal(game.board, move)) {
                            std::cout << "Before: owner=" << game.board.stoneOwner.test(idx) << "\n";
                            mover.applyMove(game.board, move);
                            std::cout << "After: owner=" << game.board.stoneOwner.test(idx) << "\n";
                            game.us = false;
                            game.aiMoved = false;
                            ma_engine_play_sound(&engine, "assets/StoneFlip.wav", NULL);
                            moveApplied = true;
                        } 
                    }
                }
            }
        } 

        if (stoneMode && game.winner == 0 && !game.awaitingBonus) {
            if (hoveredRow >= 0 && hoveredRow < gridSize && hoveredCol >= 0 && hoveredCol < gridSize) {
            int stoneIdx = stoneIndex(hoveredRow, hoveredCol);

                if (!game.board.stoneSet.test(stoneIdx)) {
                    float snapX = origin.x + (hoveredCol * cellSize);
                    float snapY = origin.y + (hoveredRow * cellSize);
                    ImVec2 snapPos(snapX, snapY);

                    ImU32 previewColor = game.board.isPlayerOne ? IM_COL32(255, 255, 255, 120) : IM_COL32(0, 0, 0, 120); 
                    drawList->AddCircleFilled(snapPos, cellSize * 0.42f, previewColor);

                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !moveApplied) {
                        Move move{game.board.isPlayerOne, hoveredRow, hoveredCol, false, MoveType::Stone};
                        if (mover.completeConnection(game.board, move) == 0) {
                            move.stoneChoices = 0;
                            if (mover.isLegal(game.board, move)) {
                                mover.applyMove(game.board, move);
                                game.us = false;
                                game.aiMoved = false;
                                ma_engine_play_sound(&engine, "assets/PlaceStone.wav", NULL);
                                moveApplied = true;
                            }
                        } else {
                            game.pendingMove = move;
                            game.awaitingBonus = true;
                            game.bonusError.clear();
                        }
                    }
                }
            }
        }

        if (stickMode && game.winner == 0 && !game.awaitingBonus) {
            if (hoveredRow >= 0 && hoveredRow < gridSize && hoveredCol >= 0 && hoveredCol < gridSize) {
                bool leanH = std::abs(offsetX) > std::abs(offsetY);
                int stickRow = -1;
                int stickCol = -1;

                if (leanH) {
                    if (offsetX > 0) {
                        stickRow = hoveredRow;
                        stickCol = hoveredCol;
                    } else if (offsetX < 0) {
                        stickRow = hoveredRow;
                        stickCol = hoveredCol - 1;
                    }
                } else {
                    if (offsetY > 0) {
                        stickRow = hoveredRow;
                        stickCol = hoveredCol;
                    } else if (offsetY < 0) {
                        stickRow = hoveredRow - 1;
                        stickCol = hoveredCol;
                    }
                }

                if (stickRow != -1 && stickCol != -1) {
                    if (leanH && stickRow >= 0 && stickRow < gridSize && stickCol >= 0 && stickCol < (gridSize - 1)) {
                        int stickIdx = hStickIndex(stickRow, stickCol);

                        if (!game.board.hStickSet.test(stickIdx)) {
                            float snapX = origin.x + (stickCol * cellSize);
                            float snapY = origin.y + (stickRow * cellSize);
                            ImVec2 snapPos(snapX, snapY);
                            ImVec2 endPos(snapX + cellSize, snapY);

                            ImU32 previewColor = game.board.isPlayerOne ? IM_COL32(255, 255, 255, 120) : IM_COL32(0, 0, 0, 120); 
                            drawList->AddLine(snapPos, endPos, previewColor, 4.0f);

                            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !moveApplied) {
                                Move move{game.board.isPlayerOne, stickRow, stickCol, true, MoveType::Stick};
                                if (mover.completeConnection(game.board, move) == 0) {
                                    move.stoneChoices = 0;
                                    if (mover.isLegal(game.board, move)) {
                                        mover.applyMove(game.board, move);
                                        game.us = false;
                                        game.aiMoved = false;
                                        ma_engine_play_sound(&engine, "assets/PlaceStick.wav", NULL);
                                        moveApplied = true;
                                    }
                                } else {
                                    game.pendingMove = move;
                                    game.awaitingBonus = true;
                                    game.bonusError.clear();
                                }
                            }
                        }
                    }
                    if (!leanH && stickRow >= 0 && stickRow < (gridSize - 1) && stickCol >= 0 && stickCol < gridSize) {
                        int stickIdx = vStickIndex(stickRow, stickCol);

                        if (!game.board.vStickSet.test(stickIdx)) {
                            float snapX = origin.x + (stickCol * cellSize);
                            float snapY = origin.y + (stickRow * cellSize);
                            ImVec2 snapPos(snapX, snapY);
                            ImVec2 endPos(snapX, snapY + cellSize);
                            
                            ImU32 previewColor = game.board.isPlayerOne ? IM_COL32(255, 255, 255, 120) : IM_COL32(0, 0, 0, 120); 
                            drawList->AddLine(snapPos, endPos, previewColor, 4.0f);

                            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !moveApplied) {
                                Move move{game.board.isPlayerOne, stickRow, stickCol, false, MoveType::Stick};
                                if (mover.completeConnection(game.board, move) == 0) {
                                    move.stoneChoices = 0;
                                    if (mover.isLegal(game.board, move)) {
                                        mover.applyMove(game.board, move);
                                        game.us = false;
                                        game.aiMoved = false;
                                        ma_engine_play_sound(&engine, "assets/PlaceStick.wav", NULL);
                                        moveApplied = true;
                                    }
                                } else {
                                    game.pendingMove = move;
                                    game.awaitingBonus = true;
                                    game.bonusError.clear();
                                }
                            }
                        }   
                    }
                }
            }
        }

        game.winner = mover.isGameOver(game.board);
        
        ImGui::Begin("Arcturus 2.0");
        if (!game.us && !game.awaitingBonus && !game.aiMoved && game.winner == 0) {
            ImGui::Text("Arcturus is thinking...");
            auto start = std::chrono::high_resolution_clock::now();
            Move bestMove = ai.findBestMove(game.board, 1000);
            auto end = std::chrono::high_resolution_clock::now();
            game.thinkTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            mover.applyMove(game.board, bestMove);
            if (bestMove.type == MoveType::Stone) {
                ma_engine_play_sound(&engine, "assets/PlaceStone.wav", NULL);
            } else if (bestMove.type == MoveType::Stick) {
                ma_engine_play_sound(&engine, "assets/PlaceStick.wav", NULL);
            } else if (bestMove.type == MoveType::Flip) {
                ma_engine_play_sound(&engine, "assets/StoneFlip.wav", NULL);
            }

            game.us = true;
            game.aiMoved = true;
        }

        if (game.thinkTime > 0 && game.thinkTime < 1000) {
            ImGui::Text("Think Time: %.2f ms",  game.thinkTime);
        } else if (game.thinkTime >= 1000) {
            ImGui::Text("Think Time: %.2f s",  game.thinkTime / 1000.0);
        }
        ImGui::Text("Eval (player to move): %.2f", ai.evaluate(game.board));
        ImGui::End();

        game.winner = mover.isGameOver(game.board);

        if (game.winner != 0) {
            ImGui::Begin("Game Over");
            ImGui::Text("Player %d wins!", game.winner);
            if (game.winner == 1 && !endSoundPlayed) {
                ma_engine_play_sound(&engine, "assets/Victory.wav", NULL);
                endSoundPlayed = true;
            } else if (game.winner == 2 && !endSoundPlayed) {
                ma_engine_play_sound(&engine, "assets/Defeat.wav", NULL);
                endSoundPlayed = true;
            }
            if (ImGui::Button("Restart")) {
                game = GameState{};
                endSoundPlayed = false;
            }
            ImGui::End();
        }

        if (game.awaitingBonus) {
            ImGui::SetNextWindowSize(ImVec2(400, 150));
            ImGui::Begin("Yield");
            int maxBonus = mover.completeConnection(game.board, game.pendingMove);
            ImGui::SliderInt("How many stones?", &game.bonusSliderValue, 0, maxBonus);
            ImGui::Text("Sticks: %d", maxBonus - game.bonusSliderValue);
            if (!game.bonusError.empty()) {
                ImGui::TextColored(
                    ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                    "%s",
                    game.bonusError.c_str());
            }
            if (ImGui::Button("Confirm")) {
                game.pendingMove.stoneChoices = game.bonusSliderValue;
                if (mover.isLegal(game.board, game.pendingMove)) {
                    mover.applyMove(game.board, game.pendingMove);
                    ma_engine_play_sound(&engine, "assets/Bonus.wav", NULL);
                    game.awaitingBonus = false;
                    game.bonusSliderValue = 0;
                    game.bonusError.clear();
                    game.us = false;
                    game.aiMoved = false;
                } else {
                    game.bonusError = "That move is no longer legal. Cancel and choose another move.";
                }
            } else if (ImGui::Button("Cancel")) {
                game.awaitingBonus = false;
                game.bonusSliderValue = 0;
                game.bonusError.clear();
            }
            ImGui::End();
        }

        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y));
        if (ImGui::Button("Place Stones")) {
            stoneMode = true;
            stickMode = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Place Sticks")) {
            stickMode = true;
            stoneMode = false;
        }

        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 20));
        ImGui::Text("Turn: %s", game.board.isPlayerOne ? "One" : "Two");
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 35));
        ImGui::Text("Player 1");
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 50));
        ImGui::Text("Points: %d", game.board.ptsOne);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 65));
        ImGui::Text("Stones: %d", game.board.stonesOne);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 80));
        ImGui::Text("Sticks: %d", game.board.sticksOne);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 105));
        ImGui::Text("Player 2");
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 120));
        ImGui::Text("Points: %d", game.board.ptsTwo);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 135));
        ImGui::Text("Stones: %d", game.board.stonesTwo);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + (gridSize * cellSize), origin.y + 150));
        ImGui::Text("Sticks: %d", game.board.sticksTwo);

        ImGui::End();
        ImGui::PopStyleColor();

        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}