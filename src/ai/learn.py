from network import canonicalizeFeatures, fromExport
import network as nt
import numpy as np
import argparse
from functools import partial
import subprocess
import os
import time
import random
import math
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

stdoutLock = Lock()

script_dir = os.path.dirname(os.path.abspath(__file__))
exe_path = os.path.abspath(os.path.join(script_dir, "..", "..", "Nexus-cli.exe"))
INITIAL_WEIGHTS_PATH = os.path.abspath(
    os.path.join(script_dir, "..", "..", "arcturus_weights_gen4.pt")
)
MODEL_PATH = os.path.abspath(
    os.path.join(script_dir, "..", "..", "arcturus_weights_gen4.onnx")
)

NEW_MODEL_PATH = os.path.abspath(
    os.path.join(script_dir, "..", "..", "arcturus_weights_gen5.onnx")
)
WEIGHTS_PATH = os.path.abspath(
    os.path.join(script_dir, "..", "..", "arcturus_weights_gen5.pt")
)
CHECKPOINT_DIR = os.path.abspath(os.path.join(script_dir, "checkpoints", "gen4"))
BUFFER_PATH = os.path.join(script_dir, "replay_buffer_14ch.pt")
HELDOUT_DATA_PATH = os.path.join(script_dir, "heldout_heuristic_14ch.pt")
NN_HOLDOUT_PATH = os.path.join(script_dir, "nn_holdout.pt")
EVALUATION_DATA_PATH = os.path.join(script_dir, "eval_comparison_14ch.pt")
PRIORS_PATH = os.path.join(script_dir, "priorsData_gen3.pt")
GAMES_LOG_PATH = os.path.abspath(os.path.join(script_dir, "..", "..", "games.txt"))

FIRST_MOVES_TO_LOG = 10
MAX_BUFFER_STATES = 50000  # Enforce rolling window cap
SAMPLE_TRAIN_SIZE = 5000  # Number of random items pulled for mini-batches
POLICY_K = 30
games_log_lock = threading.Lock()

totalMoves = 0
totalPly = 0
count = 0
unfinishedCount = 0
winOne = 0
winTwo = 0
draws = 0


def targetForPosition(winner, is_player_one_turn):
    if winner == -2:
        return -2.0 # indeterminate state
    if winner == 0:
        return 0.0
    return 1.0 if (winner == 1) == is_player_one_turn else -1.0

def parseVisitDistribution(line):
    fields = line.strip().split()
    if not fields or fields[0] != "visits" or len(fields) < 2:
        raise ValueError(f"Invalid MCTS visit distribution: {line!r}")

    distribution = {}
    for field in fields[1:]:
        try:
            action_text, probability_text = field.split(":", 1)
            action = int(action_text)
            probability = float(probability_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid MCTS visit entry: {field!r}"
            ) from error
        if action < 0 or action >= 552:
            raise ValueError(f"Policy action index out of range: {action}")
        if not torch.isfinite(torch.tensor(probability)).item() or probability < 0:
            raise ValueError(f"Invalid visit probability: {probability}")
        if action in distribution:
            raise ValueError(f"Duplicate policy action index: {action}")
        distribution[action] = probability

    total = sum(distribution.values())
    if total <= 0.0 or abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"MCTS visit probabilities must sum to 1, got {total:.9f}"
        )
    return distribution

def parsePriors(line):
    fields = line.strip().split()
    if not fields or fields[0] != "scores" or len(fields) < 2:
        raise ValueError(f"Invalid priors format: {line!r}")

    values = {}
    for field in fields[1:]:
        try:
            actionText, scoreText = field.split(":", 1)
            action = int(actionText)
            score= float(scoreText)
        except ValueError as error:
            raise ValueError(
                f"Invalid score entry: {field!r}"
            ) from error
        if action < 0 or action >= 552:
            raise ValueError(f"Policy action index out of range: {action}")
        if not torch.isfinite(torch.tensor(score)).item():
            raise ValueError(f"Invalid score: {score}")
        if action in values:
            raise ValueError(f"Duplicate policy action index: {action}")
        values[action] = score
    return values

def priorsBuilder(priorsDict, k=POLICY_K):
    if not priorsDict:
        raise RuntimeError("No priors found.")

    adjustedScores = [k * value for value in priorsDict.values()]

    high = max(adjustedScores)

    for i in range(len(adjustedScores)):
        adjustedScores[i] = math.exp(adjustedScores[i] - high)

    total = sum(adjustedScores)

    for i in range(len(adjustedScores)):
        adjustedScores[i] /= total

    return dict(zip(priorsDict.keys(), adjustedScores))

def compactPriors(sourcePath, destPath):
    if os.path.isfile(destPath):
        raise RuntimeError("Path already exists. Delete the file to overwrite.")
    if not os.path.isfile(sourcePath):
        raise RuntimeError("No source file to open.")
    data = torch.load(sourcePath, weights_only=False)
    size = len(data)
    X = torch.zeros((size, 14, 12, 12), dtype=torch.float32)
    S = torch.full((size, 552), float('nan'), dtype=torch.float32)
    gameMap = {}

    records = torch.empty(size, dtype=torch.long)

    for i in range(size):
        current = data[i]
        features = current[0]
        priors = current[1]
        game_id = current[2]
        X[i] = torch.tensor(features, dtype=torch.float32)

        if priors:
            actions = list(priors.keys())
            scores = list(priors.values())
            S[i, actions] = torch.tensor(scores, dtype=torch.float32)
        
        if game_id not in gameMap:
            gameMap[game_id] = len(gameMap)

        records[i] = gameMap[game_id]
    
    del data

    finalDict = {"X": X, "S": S, "records": records}
    torch.save(finalDict, destPath)
    print("Priors compacted and saved.")

    print(f"Number of records: {size}")
    print(f"X shape 0: {X.shape[0]}")
    print(f"Number of distinct games: {len(gameMap)}")

    mask = ~torch.isnan(S)
    counts = torch.sum(mask, dim=1)
    minMoves = counts.min().item()
    maxMoves = counts.max().item()
    print(f"Min moves: {minMoves}")
    print(f"Max moves: {maxMoves}")

def loadCompactPriors(path):
    if not os.path.isfile(path):
        raise RuntimeError("File does not exist.")
    raw = torch.load(path)
    X = raw["X"]
    S = raw["S"]
    records = raw["records"]

    XShape = X.shape[0]
    SShape = S.shape[0]
    recordsShape = records.shape[0]
    if not XShape == SShape == recordsShape:
        raise RuntimeError(f"Size mismatch | X Shape 0: {XShape} | S Shape 0: {SShape} | Records Shape 0: {recordsShape}")
    if X.shape[1:] != (14, 12, 12) or S.shape[1] != 552:
        raise RuntimeError("Mismatch in shape 1 for X or S.")

    distinctCount = torch.unique(records)
    if records.min() < 0 or distinctCount.numel() != records.max().item() + 1:
        raise RuntimeError(f"Records are out of range | Min: {records.min()} | Distinct count: {distinctCount.numel()} | Max: {records.max().item()}")
    mask = ~torch.isnan(S)
    counts = torch.sum(mask, dim=1)
    if (counts < 1).any():
        raise RuntimeError("Must have at least one non-NaN entry per row of S.")

    return X, S, records

def targetsFromScores(S, k):
    if not torch.is_tensor(S):
        raise RuntimeError("Cannot use a non-tensor.")
    
    scores = S * k
    scores[torch.isnan(scores)] = float("-inf")
    probabilities = torch.softmax(scores, dim=1)

    return probabilities

def effectiveMoves(p):
    pLogP = torch.where(p > 0, p * torch.log(p), torch.tensor(0.0))
    return torch.exp(-torch.sum(pLogP, dim=1))

def parseEvaluation(line, context):
    value_text = line.strip()
    if value_text.lower() == "null":
        return None
    try:
        value = float(value_text)
    except ValueError as error:
        raise ValueError(
            f"Invalid evaluation from {context}: {line!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite evaluation from {context}: {value_text!r}"
        )
    return value

def queryEvaluation(engine, game_id, ply, engine_name):
    context = f"{engine_name}, game {game_id}, ply {ply}"
    engine.stdin.write("eval\n")
    engine.stdin.flush()
    line = engine.stdout.readline()
    if not line:
        raise RuntimeError(f"Nexus CLI stopped while evaluating {context}")
    return parseEvaluation(line, context)

def runGame(
    id,
    model_path=MODEL_PATH,
    collect_policy_targets=False,
    collect_evaluations=False,
    collect_priors=False,
    iterations_per_move=200,
    max_plies=150,
    opening_plies=8
):
    if iterations_per_move < 1 or max_plies < 1:
        raise ValueError("Iterations per move and maximum plies must be positive")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"Nexus CLI executable does not exist: {exe_path}"
        )

    if not os.path.isfile(model_path):
        raise RuntimeError("Cannot run game with invalid model path.")
    start_player = "1" if id % 2 == 1 else "2"
    command = [exe_path]

    command.extend(
        ["--model", model_path, "--disable-policy-priors"]
    )
    command.extend(["--start-player", start_player])

    engine = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    positions = []
    policy_positions = []
    recordedPriors = []
    evaluation_positions = []
    recorded_moves = []

    ply = 0
    winner = 0
    indeterminate = False

    currentTime = time.time_ns()
    gameSeed = currentTime + id
    rng = random.Random(gameSeed)

    isOpeningPly = True

    try:
        while ply < max_plies:
            global totalPly
            totalPly += 1
            if ply > opening_plies:
                isOpeningPly = False
            
            engine.stdin.write("export\n")
            engine.stdin.flush()
            exported_position = engine.stdout.readline()
            if not exported_position:
                raise RuntimeError(
                    f"Nexus CLI stopped while exporting game {id}, ply {ply}"
                )
            features, is_player_one_turn = fromExport(exported_position)
            positions.append((features, is_player_one_turn))

            engine.stdin.write("status\n")
            engine.stdin.flush()
            status_line = engine.stdout.readline()
            if not status_line:
                raise RuntimeError(
                    f"Nexus CLI stopped while checking game {id}, ply {ply}"
                )
            winner = int(status_line)
            if winner != 0:
                break

            if collect_priors:
                engine.stdin.write("priors\n")
                engine.stdin.flush()
                priorsLine = engine.stdout.readline()
                if not priorsLine:
                    raise RuntimeError(f"Expected priors at ply {ply}")
                parsedPriors = parsePriors(priorsLine)
                recordedPriors.append((
                    features,
                    parsedPriors,
                    id
                ))

            if collect_evaluations:
                raise RuntimeError(f"Collect evaluations is no longer maintained.")

            go_command = f"go {iterations_per_move}"
            if collect_policy_targets:
                go_command += " visits"
            if ply < opening_plies:
                moveSeed = rng.randrange(1 << 31)
                engine.stdin.write(f"random {moveSeed}\n")
                engine.stdin.flush()
            else:
                engine.stdin.write(go_command + "\n")
                engine.stdin.flush()

            move_response = engine.stdout.readline().strip()
            if move_response != "moveok":
                raise RuntimeError(
                    f"Nexus CLI failed to complete go in game {id}, ply "
                    f"{ply}: {move_response!r}"
                )
            move_line = engine.stdout.readline()
            if not move_line:
                raise RuntimeError(
                    f"Nexus CLI omitted its move in game {id}, ply {ply}"
                )
            move_text = move_line.strip()
            if not move_text.startswith("Move "):
                raise RuntimeError(
                    f"Nexus CLI returned an invalid move in game {id}, "
                    f"ply {ply}: {move_text!r}"
                )
            if collect_policy_targets and not isOpeningPly:
                distribution_line = engine.stdout.readline()
                if not distribution_line:
                    raise RuntimeError(
                        f"Nexus CLI omitted visit counts for game {id}, "
                        f"ply {ply}"
                    )
                distribution = parseVisitDistribution(distribution_line)
                policy_positions.append((features, distribution, id))

            if len(recorded_moves) < FIRST_MOVES_TO_LOG:
                recorded_moves.append((ply + 1, move_text))
            ply += 1
        if winner == 0 and ply >= max_plies:
            engine.stdin.write("status\n")
            engine.stdin.flush()
            final_status = engine.stdout.readline()
            if not final_status:
                raise RuntimeError(
                    f"Nexus CLI stopped while checking the final position "
                    f"for game {id}"
                )
            winner = int(final_status)
            if winner == 0:
                indeterminate = True
    finally:
        if engine.poll() is None:
            try:
                engine.stdin.write("stop\n")
                engine.stdin.flush()
                engine.wait()
            except BrokenPipeError:   
                print("Failed to close engine with stop command | Manual shutdown...") 
                engine.stdin.close()
        
        engine.stdout.close()

    if indeterminate:
        outcome = -2.0
    else:
        if winner == 0:
            outcome = 0.0
        elif winner == 1:
            outcome = 1.0
        else:
            outcome = -1.0
        
    length = len(positions)
    with stdoutLock:
        global count, unfinishedCount, winOne, winTwo, draws
        if outcome == 1.0:
            winOne += 1
        elif outcome == -1.0:
            winTwo += 1
        elif outcome == 0.0:
            draws += 1
        if not indeterminate:
            count += 1
        else:
            unfinishedCount += 1

        length = len(positions)
        print(f"Games finished: {count} | Number of moves: {length}")
        if indeterminate:
            print(f"Game was indeterminate.")
    global totalMoves
    totalMoves += length
    value_positions = [
        (features, targetForPosition(winner, is_player_one_turn))
        for features, is_player_one_turn in positions
    ]
    evaluation_data = [
        (
            features,
            heuristic_evaluation,
            neural_evaluation,
            targetForPosition(winner, is_player_one_turn),
            game_id,
            position_ply,
        )
        for (
            features,
            heuristic_evaluation,
            neural_evaluation,
            is_player_one_turn,
            game_id,
            position_ply,
        ) in evaluation_positions
    ]

    returnDict = {}

    if collect_priors:
        returnDict["priors"] = recordedPriors

    if collect_policy_targets:
        returnDict["policy"] = policy_positions

    if collect_evaluations:
        returnDict["evaluations"] = evaluation_data

    returnDict["gameData"] = value_positions
    return returnDict

def evaluator(games,
    workers=4,
    model_path=MODEL_PATH,
    use_model=None,
    first_game_id=1,
    collect_policy_targets=False,):

    data = []
    start = time.perf_counter()
    maxWorkers = workers
    total_games = games
    model_is_active = (
        os.path.isfile(model_path) if use_model is None else use_model
    )

    opponent_description = (
        "neural-versus-heuristic"
        if model_is_active
        else "heuristic-vs-heuristic"
    )
    print(
        f"Playing {total_games} {opponent_description} games with "
        f"{maxWorkers} worker threads..."
    )
    
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        gameIds = list(range(first_game_id, first_game_id + total_games))
        run_game = partial(
            runGame,
            model_path=model_path,
            use_model=use_model,
            collect_policy_targets=collect_policy_targets,
        )
        results = executor.map(run_game, gameIds)
        
        policy_data = []
        collection_id = time.time_ns()
        for gameResult in results:
            if collect_policy_targets:
                gameData, gamePolicyData = gameResult
                data.extend(gameData)
                policy_data.extend(
                    (features, distribution, (collection_id, game_id))
                    for features, distribution, game_id in gamePolicyData
                )
            else:
                data.extend(gameResult)

    end = time.perf_counter()
    duration = end - start
    print(f"Total time collected: {duration:.2f} seconds")
    print(f"Result: {winOne}-{draws}-{winTwo}")
    print(f"Avg time per move: {duration / totalMoves if totalMoves > 0 else 0:.4f} seconds")
    if collect_policy_targets:
        return data, policy_data
    return data

def train(
    games,
    workers=4,
    model_path=MODEL_PATH,
    first_game_id=1,
    iterations=200,
    collect_policy_targets=False,
    collect_evaluations=False,
    collect_priors=False,
    opening_plies=8
):
    """Collect game data for training."""
    global count, totalMoves, winOne, winTwo, draws
    count = 0
    totalMoves = 0
    winOne = 0
    winTwo = 0
    draws = 0
    data = []
    start = time.perf_counter()
    maxWorkers = workers
    total_games = games
    if not os.path.isfile(model_path):
        raise RuntimeError("Cannot collect with invalid model path.")
    opponent_description = ("nn vs nn")
    print(f"Playing {total_games} {opponent_description} games with {maxWorkers} worker threads...")
    
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        gameIds = list(range(first_game_id, first_game_id + total_games))
        run_game = partial(
            runGame,
            model_path=model_path,
            iterations_per_move=iterations,
            collect_policy_targets=collect_policy_targets,
            collect_evaluations=collect_evaluations,
            collect_priors=collect_priors,
            opening_plies=opening_plies
        )

        results = executor.map(run_game, gameIds)
        
        policy_data = []
        evaluation_data = []
        priorsData = []
        collection_id = time.time_ns()
        for gameResult in results:
            gameData = gameResult["gameData"]
            foundIndeterminate = False
            for _, target in gameData:
                if target not in (-1, 0, 1):
                    foundIndeterminate = True
                    break
            if foundIndeterminate:
                continue
            data.extend(gameData)
            if collect_policy_targets:
                gamePolicyData = gameResult["policy"]
                policy_data.extend(
                    (features, distribution, (collection_id, game_id))
                    for features, distribution, game_id in gamePolicyData
                )
            if collect_priors:
                gamePriorsData = gameResult["priors"]
                priorsData.extend(
                    (features, priors, game_id)
                    for features, priors, game_id in gamePriorsData
                )
            if collect_evaluations:
                gameEvaluationData = gameResult["evaluations"]
                evaluation_data.extend(gameEvaluationData)

    end = time.perf_counter()
    duration = end - start
    print(f"Total time collected: {duration:.2f} seconds")
    print(f"Result: {winOne}-{draws}-{winTwo}")
    print(f"Avg time per move: {duration / totalMoves if totalMoves > 0 else 0:.4f} seconds")
    print(f"Avg ply: {totalPly / (winOne + draws + winTwo)}")
    print(f"Unfinished games: {unfinishedCount}")

    collectDict = {}
    collectDict["priors"] = priorsData
    collectDict["policy"] = policy_data
    collectDict["evaluations"] = evaluation_data
    collectDict["gameData"] = data

    return collectDict

def loadEvaluationData(path=EVALUATION_DATA_PATH):
    if not os.path.exists(path):
        return []
    stored_data = torch.load(path, weights_only=False)
    if not isinstance(stored_data, dict) or stored_data.get("version") != 1:
        raise ValueError(f"Unsupported evaluation-data format in {path}")
    records = stored_data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Evaluation records are missing or invalid in {path}")
    return [
        (
            canonicalizeFeatures(features),
            heuristic_evaluation,
            neural_evaluation,
            outcome,
            collection_id,
            game_id,
            ply,
        )
        for (
            features,
            heuristic_evaluation,
            neural_evaluation,
            outcome,
            collection_id,
            game_id,
            ply,
        ) in records
    ]

def saveEvaluationData(records, path=EVALUATION_DATA_PATH, append=True):
    if append:
        records = loadEvaluationData(path) + records
    records = records[-MAX_BUFFER_STATES:]
    temporary_path = f"{path}.tmp"
    torch.save(
        {
            "version": 1,
            "record_fields": [
                "features",
                "heuristic_eval",
                "neural_eval",
                "outcome",
                "collection_id",
                "game_id",
                "ply",
            ],
            "perspective": "side to move",
            "heuristic_eval_scale": "raw MCTS heuristic score",
            "neural_eval_scale": "bounded neural value prediction",
            "records": records,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)
    return records

def collectEval(
    games,
    workers=4,
    model_path=MODEL_PATH,
    first_game_id=1,
    output_path=EVALUATION_DATA_PATH,
):
    collection_id = time.time_ns()
    _, evaluation_data = train(
        games,
        workers,
        model_path=model_path,
        use_model=True,
        first_game_id=first_game_id,
        collect_evaluations=True,
    )
    evaluation_data = [
        (
            features,
            heuristic_evaluation,
            neural_evaluation,
            outcome,
            collection_id,
            game_id,
            ply,
        )
        for (
            features,
            heuristic_evaluation,
            neural_evaluation,
            outcome,
            game_id,
            ply,
        ) in evaluation_data
    ]
    all_records = saveEvaluationData(evaluation_data, output_path)
    print(f"Saved {len(evaluation_data)} new evaluation records to {output_path}")
    return all_records

def loadReplayBuffers():
    if os.path.exists(BUFFER_PATH):
        stored_buffer = torch.load(BUFFER_PATH, weights_only=False)
        if isinstance(stored_buffer, list):
            value_buffer = stored_buffer
            policy_buffer = []
        elif (isinstance(stored_buffer, dict) and stored_buffer.get("version") == 2):
            value_buffer = stored_buffer["value"]
            policy_buffer = stored_buffer["policy"]
        else:
            raise ValueError(f"Unsupported replay-buffer format in {BUFFER_PATH}")
        return (
            [
                (canonicalizeFeatures(features), target)
                for features, target in value_buffer
            ],
            [
                (canonicalizeFeatures(features), distribution, game_id)
                for features, distribution, game_id in policy_buffer
            ],
        )
    return [], []

def loadReplayBuffer():
    return loadReplayBuffers()[0]

def saveReplayBuffers(value_buffer, policy_buffer):
    value_buffer = value_buffer[-MAX_BUFFER_STATES:]
    policy_buffer = policy_buffer[-MAX_BUFFER_STATES:]
    temporary_path = f"{BUFFER_PATH}.tmp"
    torch.save(
        {
            "version": 2,
            "value": value_buffer,
            "policy": policy_buffer,
        },
        temporary_path,
    )
    os.replace(temporary_path, BUFFER_PATH)
    return value_buffer, policy_buffer

def exportOnnx(model, filename="arcturus_value.onnx"):
    import onnx

    dummy_input = torch.randn(1, 14, 12, 12)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['value', 'policy'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'value': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
        },
        dynamo=False
    )

    exported_model = onnx.load(filename, load_external_data=True)
    exported_model.ir_version = 9
    onnx.checker.check_model(exported_model)
    onnx.save(exported_model, filename)

    print(f"Success! Exported {filename} for C++ use.")

def exportCheckpoint(ptPath, onnxPath):
    nn = nt.Network()
    checkpoint = torch.load(ptPath, map_location="cpu")
    nn.load_state_dict(checkpoint["stateDict"])
    exportOnnx(nn, onnxPath)

def featureKey(features):
    return torch.as_tensor(features, dtype=torch.float32).contiguous().numpy().tobytes()

def unseenHoldoutData(replay_buffer, heldout_data):
    training_keys = set()
    for features, target in replay_buffer:
        if target not in (-1, 0, 1):
            raise ValueError("Game with indeterminate state found in replay buffer")
        
        training_keys.add(featureKey(canonicalizeFeatures(features)))

    unseen_data = []
    seen_holdout_keys = set()
    overlap_count = 0
    duplicate_count = 0
    for features, target in heldout_data:
        if target not in (-1, 0, 1):
            raise ValueError("Game with indeterminate state found in heldout data")
        features = canonicalizeFeatures(features)
        key = featureKey(features)
        if key in training_keys:
            overlap_count += 1
        elif key in seen_holdout_keys:
            duplicate_count += 1
        else:
            seen_holdout_keys.add(key)
            unseen_data.append((features, target))
    return unseen_data, overlap_count, duplicate_count

def evaluateHeuristicHoldout(games, workers, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Evaluation checkpoint does not exist: {checkpoint_path}"
        )

    net = nt.Network()
    net.loadModel(checkpoint_path)
    net.eval()

    print(
        f"Collecting {games} heuristic games without adding them to training "
        "or the replay buffer."
    )
    raw_data = train(
        games,
        workers,
        model_path=MODEL_PATH,
        use_model=False,
        first_game_id=1001,
    )
    gameData = raw_data["gameData"]
    for _, target in gameData:
            if target not in (-1, 0, 1):
                raise ValueError("Game with indeterminate state found in heuristic holdout data")

    torch.save(gameData, HELDOUT_DATA_PATH)
    print(f"Saved raw held-out positions to {HELDOUT_DATA_PATH}")

    replay_buffer = loadReplayBuffer()
    unseen_data, overlap_count, duplicate_count = unseenHoldoutData(
        replay_buffer,
        gameData
    )

    print(
        f"Collected {len(raw_data)} positions; excluded {overlap_count} "
        f"already-seen positions and {duplicate_count} repeated held-out "
        f"positions; evaluating {len(unseen_data)} unique unseen positions."
    )
    if not unseen_data:
        raise RuntimeError(
            "No unique unseen positions remain for held-out evaluation"
        )

    squared_error = 0.0
    total_count = 0
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(unseen_data), batch_size):
            batch = unseen_data[start:start + batch_size]
            inputs = torch.tensor(
                [features for features, _ in batch],
                dtype=torch.float32,
            )
            targets = torch.tensor(
                [target for _, target in batch],
                dtype=torch.float32,
            )
            predictions, _ = net(inputs)
            predictions = predictions.squeeze(1)
            errors = (predictions - targets) ** 2
            squared_error += errors.sum().item()
            total_count += len(batch)

    print(
        f"Held-out MSE ({total_count} unique unseen positions): "
        f"{squared_error / total_count:.6f}"
    )

def genHoldout(games=150, workers=6, iterations=200, opening_plies=8, holdoutPath=NN_HOLDOUT_PATH):
    """Generate holdout data with neural network."""
    parent_dir = os.path.dirname(holdoutPath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    print(f"Collecting {games} games for holdout data.")
    raw = train(
        games,
        workers,
        model_path=MODEL_PATH,
        first_game_id=93371,
        iterations=iterations,
        opening_plies=opening_plies
    )
    gameData = raw["gameData"]
    torch.save(gameData, holdoutPath)
    print(f"Saved heldout positions to path: {holdoutPath}")

def collectPriors(games, workers, path, model_path, iterations=200):
    if os.path.isfile(path):
        currentFile = torch.load(path)
        print(f"{path} loaded successfully.")
    else:
        currentFile = []
        print(f"Creating new dataset at {path}")
    
    rawData = train(
            games,
            workers,
            model_path=model_path,
            use_model=True,
            first_game_id=1001,
            iterations=iterations,
            collect_priors=True
    )

    priors = rawData["priors"]
    currentFile.extend(priors)
    torch.save(currentFile, path)
    print(f"Priors saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-heuristic",
        action="store_true",
        help="collect a separate heuristic holdout set and evaluate checkpoint MSE",
    )
    parser.add_argument(
        "--collect-evals",
        action="store_true",
        help="collect heuristic and neural evals for the same positions",
    )
    parser.add_argument("--collect", default=False, action="store_true", help="collect data through self-play")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--checkpoint", default=WEIGHTS_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", default=EVALUATION_DATA_PATH)
    parser.add_argument("--collect-priors", action="store_true")
    parser.add_argument("--offline", action="store_true", help="train model on replay buffer without self-play")
    parser.add_argument("--holdout", action="store_true", help="generate holdout data")
    parser.add_argument("--no-train", action="store_true", help="toggle train in collection")
    args = parser.parse_args()

    if args.evaluate_heuristic:
        evaluateHeuristicHoldout(
            args.games,
            args.workers,
            os.path.abspath(args.checkpoint),
        )
        raise SystemExit(0)

    if args.holdout:
        defaultGames = 150 if args.games == 60 else args.games
        genHoldout(
            games=defaultGames,
            workers=args.workers,
            iterations=int(args.iterations),
            holdoutPath=NN_HOLDOUT_PATH
        )
        raise SystemExit(0)
    
    if args.collect_evals:
        collectEval(
            args.games,
            args.workers,
            model_path=os.path.abspath(args.model),
            output_path=os.path.abspath(args.output),
        )
        raise SystemExit(0)

    if args.collect_priors:
        collectPriors(
            games=args.games,
            workers=args.workers, 
            path=PRIORS_PATH, 
            model_path=os.path.abspath(args.model), 
            iterations=args.iterations)
        raise SystemExit(0)

    if args.offline:
        net = nt.Network() 
        net.loadModel(INITIAL_WEIGHTS_PATH)
    
        replayBuffer, policyReplayBuffer = loadReplayBuffers()
        saveReplayBuffers(replayBuffer, policyReplayBuffer)

        if not replayBuffer:
            raise RuntimeError(
                f"No value positions found in replay buffer: {BUFFER_PATH}"
            )
        trainingBatch = random.sample(replayBuffer, len(replayBuffer))
        for _, target in trainingBatch:
                if target not in (-1, 0, 1):
                    raise ValueError("Game with indeterminate state found in training batch")
        if not os.path.isfile(NN_HOLDOUT_PATH):
            raise FileNotFoundError(
                f"NN holdout data is required for validation. Expected it at: {NN_HOLDOUT_PATH}"
            )
        heldoutData = torch.load(NN_HOLDOUT_PATH, weights_only=False)
        validationData, overlapCount, duplicateCount = unseenHoldoutData(
            replayBuffer,
            heldoutData
        )
        if not validationData:
            raise RuntimeError(
                "No unique held-out positions remain after excluding replay-buffer overlap"
            )
        for _, target in validationData:
            if target not in (-1, 0, 1):
                raise ValueError("Game with indeterminate state found in validation data")
        print(f"Randomly sampled {len(trainingBatch)} current-player-relative training positions.")
        print(f"Validation set: {len(validationData)} unseen positions "
                f"({overlapCount} replay overlaps and {duplicateCount} duplicates excluded)."
        )

        net.trainModel(
            trainingBatch,
            6,
            64,
            save_path=WEIGHTS_PATH,
            validation_data=validationData,
            policy_data=None,
            policy_validation_data=None,
            checkpoint_dir=CHECKPOINT_DIR,
            restore_best=True,
            train_trunk=True,
            stopEarly=True
        )
        exportOnnx(net, filename=NEW_MODEL_PATH)

        raise SystemExit(0)
    
    if args.collect:
        net = nt.Network() 
        net.loadModel(INITIAL_WEIGHTS_PATH)

        replayBuffer, policyReplayBuffer = loadReplayBuffers()

        rawData = train(games=args.games,
        workers=args.workers,
        model_path=args.model, 
        iterations=args.iterations,
        opening_plies=8
        )

        replayBuffer.extend(rawData["gameData"])
        saveReplayBuffers(replayBuffer, policyReplayBuffer)

        if not replayBuffer:
            raise RuntimeError(
                f"No value positions found in replay buffer: {BUFFER_PATH}"
            )

        if args.no_train:
            print("Skipped training model.")
            raise SystemExit(0)
        
        trainingBatch = random.sample(replayBuffer, len(replayBuffer))
        for _, target in trainingBatch:
            if target not in (-1, 0, 1):
                raise ValueError("Game with indeterminate state found in training batch")
        if not os.path.isfile(NN_HOLDOUT_PATH):
            raise FileNotFoundError(
                f"NN holdout data is required for validation. Expected it at: {NN_HOLDOUT_PATH}"
            )
        heldoutData = torch.load(NN_HOLDOUT_PATH, weights_only=False)
        validationData, overlapCount, duplicateCount = unseenHoldoutData(
            replayBuffer,
            heldoutData
        )
        if not validationData:
            raise RuntimeError(
                "No unique held-out positions remain after excluding replay-buffer overlap"
            )
        for _, target in validationData:
            if target not in (-1, 0, 1):
                raise ValueError("Game with indeterminate state found in validation data")
        print(f"Randomly sampled {len(trainingBatch)} current-player-relative training positions.")
        print(f"Validation set: {len(validationData)} unseen positions "
                f"({overlapCount} replay overlaps and {duplicateCount} duplicates excluded)."
        )

        net.trainModel(
            trainingBatch,
            6,
            64,
            save_path=WEIGHTS_PATH,
            validation_data=validationData,
            policy_data=None,
            policy_validation_data=None,
        )
        exportOnnx(net, filename=NEW_MODEL_PATH)
