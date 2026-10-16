#pragma once

#include <string>
#include <vector>

#include "onnxruntime_c_api.h"
#include "../board.h"

class InferenceRuntime {
public:
    InferenceRuntime();
    ~InferenceRuntime();

    InferenceRuntime(const InferenceRuntime&) = delete;
    InferenceRuntime& operator=(const InferenceRuntime&) = delete;

    const OrtApi* api() const { return api_; }
    OrtEnv* env() const { return env_; }

private:
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
};

class InferenceEngine {
    public:
        ~InferenceEngine();
        InferenceEngine() = default;
        InferenceEngine(const InferenceEngine&) = delete;
        InferenceEngine& operator=(const InferenceEngine&) = delete;

        bool load(const std::string& modelPath, const InferenceRuntime& runtime);

        bool hasPolicyHead() const { return hasPolicyHead_; }
        float predict(const Board& board);
        std::vector<float> predictPolicy(const Board& board);
        std::vector<float> predictBatch(const std::vector<Board>& boards);
    private:
        const OrtApi* api = nullptr;
        OrtEnv* env = nullptr;
        OrtSession* session = nullptr;
        OrtMemoryInfo* memoryInfo = nullptr;
        bool loaded = false;
        bool hasPolicyHead_ = false;
};
