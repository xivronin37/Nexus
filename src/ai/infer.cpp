
#include <iostream>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "onnxruntime_c_api.h"
#include "infer.h"

namespace {
constexpr uint32_t kOnnxRuntimeApiVersion = 17;
constexpr std::size_t kFeatureChannels = 14;
constexpr std::size_t kBoardWidth = 12;
constexpr std::size_t kBoardArea = kBoardWidth * kBoardWidth;
constexpr std::size_t kInputElements = kFeatureChannels * kBoardArea;
constexpr std::size_t kPolicyActions = 552;

std::array<float, kInputElements> encodeBoard(const Board& board) {
    std::array<float, kInputElements> encoded{};

    for (std::size_t index = 0; index < board.stoneSet.size(); ++index) {
        const std::size_t row = index / kBoardWidth;
        const std::size_t col = index % kBoardWidth;
        const std::size_t position = row * kBoardWidth + col;
        encoded[position] = board.stoneSet.test(index) ? 1.0f : 0.0f;
        encoded[kBoardArea + position] =
            board.stoneSet.test(index) &&
                    (board.stoneOwner.test(index) == board.isPlayerOne)
                ? 1.0f
                : 0.0f;
    }

    for (std::size_t index = 0; index < board.hStickSet.size(); ++index) {
        const std::size_t row = index / 11;
        const std::size_t col = index % 11;
        const std::size_t horizontalPosition = row * kBoardWidth + col;
        encoded[2 * kBoardArea + horizontalPosition] = board.hStickSet.test(index) ? 1.0f : 0.0f;
        encoded[3 * kBoardArea + horizontalPosition] =
            board.hStickSet.test(index) &&
                    (board.hStickOwner.test(index) == board.isPlayerOne)
                ? 1.0f
                : 0.0f;

        const std::size_t verticalRow = index % 11;
        const std::size_t verticalCol = index / 11;
        const std::size_t verticalPosition =
            verticalRow * kBoardWidth + verticalCol;
        encoded[4 * kBoardArea + verticalPosition] = board.vStickSet.test(index) ? 1.0f : 0.0f;
        encoded[5 * kBoardArea + verticalPosition] =
            board.vStickSet.test(index) &&
                    (board.vStickOwner.test(index) == board.isPlayerOne)
                ? 1.0f
                : 0.0f;
    }

    for (std::size_t index = 0; index < kBoardArea; ++index) {
        encoded[6 * kBoardArea + index] = 1.0f;
    }

    const std::array<float, 6> scalarFeatures = {
        static_cast<float>(
            board.isPlayerOne ? board.stonesOne : board.stonesTwo) / 10.0f,
        static_cast<float>(
            board.isPlayerOne ? board.sticksOne : board.sticksTwo) / 10.0f,
        static_cast<float>(
            board.isPlayerOne ? board.ptsOne : board.ptsTwo) / 30.0f,
        static_cast<float>(
            board.isPlayerOne ? board.stonesTwo : board.stonesOne) / 10.0f,
        static_cast<float>(
            board.isPlayerOne ? board.sticksTwo : board.sticksOne) / 10.0f,
        static_cast<float>(
            board.isPlayerOne ? board.ptsTwo : board.ptsOne) / 30.0f
    };
    for (std::size_t scalarIndex = 0;
         scalarIndex < scalarFeatures.size();
         ++scalarIndex) {
        const std::size_t channel = 7 + scalarIndex;
        for (std::size_t index = 0; index < kBoardArea; ++index) {
            encoded[channel * kBoardArea + index] =
                scalarFeatures[scalarIndex];
        }
    }

    for (std::size_t age = 0; age < board.recentlyFlipped.size(); ++age) {
        const int flippedIndex = board.recentlyFlipped[age];
        if (flippedIndex < 0 ||
            static_cast<std::size_t>(flippedIndex) >= kBoardArea) {
            throw std::runtime_error(
                "Board contains an invalid recently-flipped index");
        }
        encoded[13 * kBoardArea +
                static_cast<std::size_t>(flippedIndex)] =
            static_cast<float>(age + 1) / 6.0f;
    }
    return encoded;
}

std::string takeOrtError(const OrtApi* api, OrtStatus* status) {
    const char* message = api->GetErrorMessage(status);
    std::string error = message != nullptr ? message : "Unknown ONNX Runtime error";
    api->ReleaseStatus(status);
    return error;
}

struct BatchValues {
    const OrtApi* api;
    OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;

    ~BatchValues() {
        if (outputInfo != nullptr) {
            api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        }
        if (output != nullptr) {
            api->ReleaseValue(output);
        }
        if (input != nullptr) {
            api->ReleaseValue(input);
        }
    }
};
}

InferenceRuntime::InferenceRuntime() {
    api_ = OrtGetApiBase()->GetApi(kOnnxRuntimeApiVersion);
    if (api_ == nullptr) {
        throw std::runtime_error("Failed to get ONNX Runtime API");
    }

    OrtStatus* status = api_->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING, "NexusInference", &env_);
    if (status != nullptr) {
        throw std::runtime_error(
            "CreateEnv failed: " + takeOrtError(api_, status));
    }
}

InferenceRuntime::~InferenceRuntime() {
    if (env_ != nullptr) {
        api_->ReleaseEnv(env_);
    }
}

InferenceEngine::~InferenceEngine() {
    OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;
    if (api != nullptr && input != nullptr) {
        api->ReleaseValue(input);
    }
    if (api != nullptr && output != nullptr) {
        api->ReleaseValue(output);
    }
    if (api != nullptr && outputInfo != nullptr) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
    }
    if (session != nullptr) {
        api->ReleaseSession(session);
    }
    if (memoryInfo != nullptr) {
        api->ReleaseMemoryInfo(memoryInfo);
    }
}

bool InferenceEngine::load(
    const std::string& modelPath,
    const InferenceRuntime& runtime) {
    loaded = false;

    if (modelPath.empty()) {
        std::cerr << "Model path is empty" << std::endl;
        return false;
    }

    if (session != nullptr) {
        std::cerr << "InferenceEngine already has a loaded model" << std::endl;
        return false;
    }

    api = runtime.api();
    env = runtime.env();

    const std::filesystem::path path = std::filesystem::u8path(modelPath);
    std::error_code pathError;
    if (!std::filesystem::is_regular_file(path, pathError)) {
        std::cerr << "Model path is not a regular file: " << modelPath;
        if (pathError) {
            std::cerr << " (" << pathError.message() << ")";
        }
        std::cerr << std::endl;
        return false;
    }

    OrtSessionOptions* options = nullptr;
    OrtStatus* status = api->CreateSessionOptions(&options);
    if (status != nullptr) {
        std::cerr << "CreateSessionOptions failed: "
                  << takeOrtError(api, status) << std::endl;
        return false;
    }

    status = api->CreateSession(env, path.c_str(), options, &session);
    api->ReleaseSessionOptions(options);
    if (status != nullptr) {
        std::cerr << "CreateSession failed: "
                  << takeOrtError(api, status) << std::endl;
        return false;
    }

    OrtAllocator* allocator = nullptr;
    status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        std::cerr << "GetAllocatorWithDefaultOptions failed: "
                  << takeOrtError(api, status) << std::endl;
        api->ReleaseSession(session);
        session = nullptr;
        return false;
    }

    size_t outputCount = 0;
    status = api->SessionGetOutputCount(session, &outputCount);
    if (status != nullptr) {
        std::cerr << "SessionGetOutputCount failed: "
                  << takeOrtError(api, status) << std::endl;
        api->ReleaseSession(session);
        session = nullptr;
        return false;
    }
    for (size_t index = 0; index < outputCount; ++index) {
        char* outputName = nullptr;
        status = api->SessionGetOutputName(
            session, index, allocator, &outputName);
        if (status != nullptr) {
            std::cerr << "SessionGetOutputName failed: "
                      << takeOrtError(api, status) << std::endl;
            api->ReleaseSession(session);
            session = nullptr;
            return false;
        }
        if (outputName != nullptr && std::strcmp(outputName, "policy") == 0) {
            hasPolicyHead_ = true;
        }
        if (outputName != nullptr) {
            api->AllocatorFree(allocator, outputName);
        }
    }

    loaded = true;
    return true;
}

float InferenceEngine::predict(const Board& board) {
    if (!loaded || session == nullptr) {
        throw std::runtime_error("InferenceEngine is not loaded");
    }
    OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;

    std::array<float, kInputElements> inputData = encodeBoard(board);

    if (memoryInfo == nullptr) {
        OrtStatus* memoryStatus = api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
        if (memoryStatus != nullptr) {
            throw std::runtime_error(
                "CreateCpuMemoryInfo failed: " +
                takeOrtError(api, memoryStatus));
        }
    }

    const int64_t shape[] = {1, 14, 12, 12};
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        memoryInfo,
        inputData.data(),
        sizeof(inputData),
        shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input);
    if (status != nullptr) {
        throw std::runtime_error(
            "CreateTensorWithDataAsOrtValue failed: " +
            takeOrtError(api, status));
    }

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"value"};
    const OrtValue* inputTensors[] = {input};
    status = api->Run(
        session, nullptr, inputNames, inputTensors, 1,
        outputNames, 1, &output);

    api->ReleaseValue(input);
    input = nullptr;
    if (status != nullptr) {
        if (output != nullptr) {
            api->ReleaseValue(output);
            output = nullptr;
        }
        throw std::runtime_error("Run failed: " + takeOrtError(api, status));
    }

    status = api->GetTensorTypeAndShape(output, &outputInfo);
    if (status != nullptr) {
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error(
            "GetTensorTypeAndShape failed: " + takeOrtError(api, status));
    }

    ONNXTensorElementDataType outputType;
    status = api->GetTensorElementType(outputInfo, &outputType);
    if (status != nullptr) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error(
            "GetTensorElementType failed: " + takeOrtError(api, status));
    }
    if (outputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error("Model output is not a float tensor");
    }

    size_t outputElementCount = 0;
    status = api->GetTensorShapeElementCount(outputInfo, &outputElementCount);
    if (status != nullptr) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error(
            "GetTensorShapeElementCount failed: " +
            takeOrtError(api, status));
    }
    if (outputElementCount != 1) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error(
            "Model output must contain exactly one float");
    }

    void* outputData = nullptr;
    status = api->GetTensorMutableData(output, &outputData);
    if (status != nullptr) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error(
            "GetTensorMutableData failed: " + takeOrtError(api, status));
    }
    if (outputData == nullptr) {
        api->ReleaseTensorTypeAndShapeInfo(outputInfo);
        outputInfo = nullptr;
        api->ReleaseValue(output);
        output = nullptr;
        throw std::runtime_error("Model returned a null output buffer");
    }

    const float result = *static_cast<float*>(outputData);
    api->ReleaseTensorTypeAndShapeInfo(outputInfo);
    outputInfo = nullptr;
    api->ReleaseValue(output);
    output = nullptr;
    if (!std::isfinite(result)) {
        throw std::runtime_error("Model returned a non-finite prediction");
    }
    return result;
}

std::vector<float> InferenceEngine::predictPolicy(const Board& board) {
    if (!loaded || session == nullptr) {
        throw std::runtime_error("InferenceEngine is not loaded");
    }

    OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;

    std::array<float, kInputElements> inputData = encodeBoard(board);
    if (memoryInfo == nullptr) {
        OrtStatus* memoryStatus = api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
        if (memoryStatus != nullptr) {
            throw std::runtime_error(
                "CreateCpuMemoryInfo failed: " +
                takeOrtError(api, memoryStatus));
        }
    }

    BatchValues values{api};
    const int64_t shape[] = {1, 14, 12, 12};
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        memoryInfo,
        inputData.data(),
        sizeof(inputData),
        shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &values.input);
    if (status != nullptr) {
        throw std::runtime_error(
            "CreateTensorWithDataAsOrtValue failed: " +
            takeOrtError(api, status));
    }

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"policy"};
    const OrtValue* inputValues[] = {values.input};
    status = api->Run(
        session,
        nullptr,
        inputNames,
        inputValues,
        1,
        outputNames,
        1,
        &values.output);
    if (status != nullptr) {
        throw std::runtime_error(
            "Policy inference failed: " + takeOrtError(api, status));
    }

    status = api->GetTensorTypeAndShape(values.output, &values.outputInfo);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorTypeAndShape failed: " + takeOrtError(api, status));
    }

    ONNXTensorElementDataType outputType;
    status = api->GetTensorElementType(values.outputInfo, &outputType);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorElementType failed: " + takeOrtError(api, status));
    }
    if (outputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Policy output is not a float tensor");
    }

    size_t outputElementCount = 0;
    status = api->GetTensorShapeElementCount(
        values.outputInfo, &outputElementCount);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorShapeElementCount failed: " +
            takeOrtError(api, status));
    }
    if (outputElementCount != kPolicyActions) {
        throw std::runtime_error(
            "Policy output must contain exactly 552 action logits");
    }

    void* outputData = nullptr;
    status = api->GetTensorMutableData(values.output, &outputData);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorMutableData failed: " + takeOrtError(api, status));
    }
    if (outputData == nullptr) {
        throw std::runtime_error("Model returned a null policy buffer");
    }

    const float* logits = static_cast<const float*>(outputData);
    std::vector<float> result(logits, logits + outputElementCount);
    for (float logit : result) {
        if (!std::isfinite(logit)) {
            throw std::runtime_error(
                "Model returned a non-finite policy logit");
        }
    }
    return result;
}

std::vector<float> InferenceEngine::predictBatch(
    const std::vector<Board>& boards) {
    if (!loaded || session == nullptr) {
        throw std::runtime_error("InferenceEngine is not loaded");
    }
    if (boards.empty()) {
        throw std::invalid_argument("Cannot predict an empty batch");
    }
    if (boards.size() >
            std::numeric_limits<std::size_t>::max() / kInputElements ||
        boards.size() > static_cast<std::size_t>(
                            std::numeric_limits<int64_t>::max())) {
        throw std::length_error("Inference batch is too large");
    }

    OrtValue* input = nullptr;
    OrtValue* output = nullptr;
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;

    std::vector<float> inputData(boards.size() * kInputElements);
    for (std::size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex) {
        const auto encoded = encodeBoard(boards[boardIndex]);
        std::copy(
            encoded.begin(),
            encoded.end(),
            inputData.begin() + boardIndex * kInputElements);
    }

    if (memoryInfo == nullptr) {
        OrtStatus* status = api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
        if (status != nullptr) {
            throw std::runtime_error(
                "CreateCpuMemoryInfo failed: " +
                takeOrtError(api, status));
        }
    }

    BatchValues values{api};
    const int64_t shape[] = {
        static_cast<int64_t>(boards.size()),
        14,
        12,
        12
    };
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        memoryInfo,
        inputData.data(),
        inputData.size() * sizeof(float),
        shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &values.input);
    if (status != nullptr) {
        throw std::runtime_error(
            "CreateTensorWithDataAsOrtValue failed: " +
            takeOrtError(api, status));
    }

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"value"};
    const OrtValue* inputValues[] = {values.input};
    status = api->Run(
        session,
        nullptr,
        inputNames,
        inputValues,
        1,
        outputNames,
        1,
        &values.output);
    if (status != nullptr) {
        throw std::runtime_error(
            "Run failed: " + takeOrtError(api, status));
    }

    status = api->GetTensorTypeAndShape(values.output, &values.outputInfo);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorTypeAndShape failed: " +
            takeOrtError(api, status));
    }

    ONNXTensorElementDataType outputType;
    status = api->GetTensorElementType(values.outputInfo, &outputType);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorElementType failed: " +
            takeOrtError(api, status));
    }
    if (outputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Model output is not a float tensor");
    }

    size_t outputElementCount = 0;
    status = api->GetTensorShapeElementCount(
        values.outputInfo, &outputElementCount);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorShapeElementCount failed: " +
            takeOrtError(api, status));
    }
    if (outputElementCount != boards.size()) {
        throw std::runtime_error(
            "Model output count does not match inference batch size");
    }

    void* outputData = nullptr;
    status = api->GetTensorMutableData(values.output, &outputData);
    if (status != nullptr) {
        throw std::runtime_error(
            "GetTensorMutableData failed: " +
            takeOrtError(api, status));
    }
    if (outputData == nullptr) {
        throw std::runtime_error("Model returned a null output buffer");
    }

    const float* outputValues = static_cast<const float*>(outputData);
    std::vector<float> predictions(
        outputValues, outputValues + outputElementCount);
    for (float prediction : predictions) {
        if (!std::isfinite(prediction)) {
            throw std::runtime_error(
                "Model returned a non-finite prediction");
        }
    }
    return predictions;
}
