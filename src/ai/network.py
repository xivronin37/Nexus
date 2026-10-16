import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from tools import constantPredictor, evalModelPerformance


class Network(nn.Module):
    def __init__(self, hidden_size=128, numActions=552):
        super().__init__()

        self.input = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.l1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(32, 4, kernel_size=1)

        # value head
        self.l3 = nn.Linear(576, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        # policy head
        self.p1 = nn.Linear(576, hidden_size)
        self.p2 = nn.Linear(hidden_size, numActions)


    def forward(self, x):
        features = self._sharedFeatures(x)
        return self._headsFromFeatures(features)

    def _sharedFeatures(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x.flatten(start_dim=1)

    def _headsFromFeatures(self, features):
        value = torch.tanh(self.l4(F.relu(self.l3(features))))
        policy_logits = self.p2(F.relu(self.p1(features)))
        return value, policy_logits

    def _validationValueMse(self, loader):
        self.eval()
        squaredError = 0.0
        count = 0
        with torch.no_grad():
            for inputs, results in loader:
                outputs, _ = self.forward(inputs)
                squaredError += torch.sum((outputs - results) ** 2).item()
                count += results.numel()
        return squaredError / count

    def _validationPolicyMetrics(self, loader):
        self.eval()
        lossTotal = 0.0
        count = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets, legalMask in loader:
                _, logits = self.forward(inputs)
                maskedLogits = logits.masked_fill(
                    ~legalMask,
                    torch.finfo(logits.dtype).min,
                )
                logProbabilities = F.log_softmax(maskedLogits, dim=1)
                lossTotal += -(
                    targets * logProbabilities
                ).sum().item()
                count += targets.shape[0]
                correct += (
                    maskedLogits.argmax(dim=1) == targets.argmax(dim=1)
                ).sum().item()
        return lossTotal / count, correct / count

    def trainModel(
        self,
        data,
        epochs=10,
        batchSize=64,
        save_path="arcturusv2_weights.pt",
        validation_data=None,
        patience=5,
        weight_decay=1e-4,
        policy_data=None,
        policy_validation_data=None,
        policy_loss_weight=0.1,
        checkpoint_dir=None,
        restore_best=True,
        train_trunk=True,
        stopEarly=True
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if policy_loss_weight < 0:
            raise ValueError("policy_loss_weight must be non-negative")
        if train_trunk and policy_data is not None:
            raise RuntimeError("Cannot train trunk with policy data provided.")
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                print("Directory does not exist. Creating directory...")
            else:
                print("Directory exists. Clearing existing checkpoints.")
                with os.scandir(checkpoint_dir) as filenames:
                    for file in filenames:
                        if file.is_file():
                            os.remove(file.path)
                print("Checkpoints cleared.")

        print(f"Training network for {epochs} epochs with batch size {batchSize}...")
        rawInputs = torch.tensor([sample[0] for sample in data], dtype=torch.float32)
        rawResults = torch.tensor([sample[1] for sample in data], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(rawInputs, rawResults)
        dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        validationLoader = None
        if validation_data:
            validationInputs = torch.tensor(
                [sample[0] for sample in validation_data],
                dtype=torch.float32,
            )
            validationResults = torch.tensor(
                [sample[1] for sample in validation_data],
                dtype=torch.float32,
            ).unsqueeze(1)
            validationLoader = DataLoader(
                TensorDataset(validationInputs, validationResults),
                batch_size=batchSize,
                shuffle=False,
            )

        if validationLoader is not None:
            trainTargets = []
            valTargets = []
            for _, results in dataLoader:
                trainTargets.append(results.detach().cpu())
            for _, results in validationLoader:
                valTargets.append(results.detach().cpu())
            flatTrainTargets = torch.cat(trainTargets).reshape(-1)
            flatValTargets = torch.cat(valTargets).reshape(-1)

            constantData = constantPredictor(flatTrainTargets, flatValTargets)
            print("="*30)
            print(f"cTrain: {constantData["cTrain"]:.5f}")
            print(f"cVal: {constantData["cVal"]:.5f}")
            print(f"oracleMse: {constantData["oracleMse"]:.5f}")
            print(f"fairMse: {constantData["fairMse"]:.5f}")
            print(f"distShift: {constantData["distShift"]:.5f}")
            print("="*30)



        policyLoader = None
        if policy_data:
            policyInputs, policyTargets, policyLegalMask = self._policyTensors(
                policy_data,
                "training",
            )
            policyLoader = DataLoader(
                TensorDataset(policyInputs, policyTargets, policyLegalMask),
                batch_size=batchSize,
                shuffle=True,
            )

        policyValidationLoader = None
        if policy_validation_data:
            (
                policyValidationInputs,
                policyValidationTargets,
                policyValidationLegalMask,
            ) = self._policyTensors(policy_validation_data, "validation")
            policyValidationLoader = DataLoader(
                TensorDataset(
                    policyValidationInputs,
                    policyValidationTargets,
                    policyValidationLegalMask,
                ),
                batch_size=batchSize,
                shuffle=False,
            )
        param_groups = [
            {
                "params": list(self.l3.parameters()) + list(self.l4.parameters()),
                "lr": 0.001, # Value learning rate
                "weight_decay": weight_decay
            }
        ]

        if train_trunk:
            param_groups.append({
                "params": (
                    list(self.input.parameters()) + list(self.l1.parameters()) + list(self.l2.parameters())
                ),
                "lr": 3e-4, # Trunk learning rate
                "weight_decay": weight_decay
            })

        valueOptimizer = optim.Adam(param_groups)

        policyOptimizer = optim.Adam(
            list(self.p1.parameters()) + list(self.p2.parameters()),
            lr=0.001,
            weight_decay=weight_decay,
        )

        valueScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            valueOptimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
        policyScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            policyOptimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )

        criterion = nn.MSELoss()
        if not train_trunk:
            for parameter in (list(self.input.parameters()) + list(self.l1.parameters()) + list(self.l2.parameters())):
                parameter.requires_grad_(False)
        else:
            for parameter in (list(self.input.parameters()) + list(self.l1.parameters()) + list(self.l2.parameters())):
                parameter.requires_grad_(True)

        trunkParams = sum(
            p.numel() for group in valueOptimizer.param_groups for p in group["params"]
        )

        print(f"Trainable parameters in trunk: {trunkParams:,}")

        def modelState():
            return {
                name: parameter.detach().clone()
                for name, parameter in self.state_dict().items()
            }
        
        def valueHeadState():
            return {
                name: parameter.detach().clone()
                for name, parameter in self.state_dict().items()
                if name.startswith(("l3.", "l4."))
            }

        def policyHeadState():
            return {
                name: parameter.detach().clone()
                for name, parameter in self.state_dict().items()
                if name.startswith(("p1.", "p2."))
            }

        bestValueMse = (
            self._validationValueMse(validationLoader)
            if validationLoader is not None
            else float("inf")
        )
        bestValueState = modelState()
        valueUpdated = False
        policyUpdated = False
        bestPolicyLoss = (
            self._validationPolicyMetrics(policyValidationLoader)[0]
            if policyValidationLoader is not None
            else float("inf")
        )
        bestPolicyState = policyHeadState()
        epochsWithoutValueImprovement = 0
        epochsWithoutPolicyImprovement = 0
        bestEpoch = 0
        self.train()

        for epoch in range(epochs):
            runningValueLoss = 0.0
            runningPolicyLoss = 0.0
            policyUpdateCount = 0
            for inputs, results in dataLoader:
                valueOptimizer.zero_grad()
                if train_trunk:
                    features = self._sharedFeatures(inputs)
                else:
                    with torch.no_grad():
                        features = self._sharedFeatures(inputs)
                valueOutputs, _ = self._headsFromFeatures(features)
                valueLoss = criterion(valueOutputs, results)
                valueLoss.backward()
                valueOptimizer.step()
                runningValueLoss += valueLoss.item()

            if policyLoader is not None:
                for policyInputs, policyTargets, policyLegalMask in policyLoader:
                    policyOptimizer.zero_grad()
                    with torch.no_grad():
                        features = self._sharedFeatures(policyInputs)
                    _, policyLogits = self._headsFromFeatures(features)
                    policyLoss = self._softPolicyLoss(
                        policyLogits,
                        policyTargets,
                        policyLegalMask,
                    )
                    (policy_loss_weight * policyLoss).backward()
                    policyOptimizer.step()
                    runningPolicyLoss += policyLoss.item()
                    policyUpdateCount += 1
            epoch_message = (
                f"Epoch {epoch+1}/{epochs}, "
                f"Avg Training MSE: {runningValueLoss / len(dataLoader):.5f}"
            )
            if policyLoader is not None:
                epoch_message += (
                    f", Avg Policy Loss: "
                    f"{runningPolicyLoss / policyUpdateCount:.5f} "
                    f"({policyUpdateCount} policy batches)"
                )

            validationMse = None
            if validationLoader is not None:
                validationMse = self._validationValueMse(validationLoader)
                epoch_message += f", Validation MSE: {validationMse:.5f}"
                fairMse = constantData["fairMse"]
                evalDict = evalModelPerformance(validationMse, fairMse)
                epoch_message += f", Ratio: {evalDict["ratio"]}, R^2: {evalDict["fractionalImprovement"]}"

            validationPolicyLoss = None
            if policyValidationLoader is not None:
                validationPolicyLoss, policyAccuracy = (
                    self._validationPolicyMetrics(policyValidationLoader)
                )
                epoch_message += (
                    f", Validation Policy Loss: {validationPolicyLoss:.5f}, "
                    f"Policy Top-1: {policyAccuracy:.3%}"
                )

            if validationMse is not None:
                valueScheduler.step(validationMse)
                if validationMse < bestValueMse:
                    bestValueMse = validationMse
                    bestValueState = modelState()
                    valueUpdated = True
                    bestEpoch = epoch + 1
                    epochsWithoutValueImprovement = 0
                else:
                    epochsWithoutValueImprovement += 1
            if validationPolicyLoss is not None:
                policyScheduler.step(validationPolicyLoss)
                if validationPolicyLoss < bestPolicyLoss:
                    bestPolicyLoss = validationPolicyLoss
                    bestPolicyState = policyHeadState()
                    policyUpdated = True
                    epochsWithoutPolicyImprovement = 0
                else:
                    epochsWithoutPolicyImprovement += 1

            if checkpoint_dir is not None:
                epochDict = {
                    "stateDict": self.state_dict(),
                    "epoch": epoch + 1,
                    "validationMse": validationMse,
                    "validationPolicyLoss": validationPolicyLoss
                }
                epochFileName = f"e{epoch+1}.pt"
                final = os.path.join(checkpoint_dir, epochFileName)
                temp = os.path.join(checkpoint_dir, f"{epochFileName}.tmp")

                try:
                    torch.save(epochDict, temp)
                    os.replace(temp, final)  
                except Exception as e:
                    if os.path.exists(temp):
                        os.remove(temp)
                    raise RuntimeError(f"Failed to write checkpoint cleanly for epoch {epoch + 1}: {e}") from e
            
            self.train()
            print(epoch_message)
            flags = []
            valueActive = (epochsWithoutValueImprovement >= patience)
            policyActive = (epochsWithoutPolicyImprovement >= patience)

            if validationMse is not None:
                flags.append(valueActive)

            if validationPolicyLoss is not None:
                flags.append(policyActive)
            
            if flags and all(flags) and stopEarly:
                print(
                    f"Early stopping after {epoch + 1} epochs; neither "
                    "head improved its own validation metric."
                )
                break
        if restore_best:
            if valueUpdated:
                self.load_state_dict(bestValueState)
            if policyUpdated:
                self.load_state_dict(bestPolicyState)
            if validationLoader is not None:
                self.load_state_dict(bestValueState)
                print(
                    f"Restored best value head at epoch {bestEpoch}; validation MSE: {bestValueMse:.5f}."
                    if bestEpoch > 0 else f"Restored initial weights."
                )
            if policyValidationLoader is not None:
                print(
                    f"Restored best policy head; validation loss: "
                    f"{bestPolicyLoss:.5f}."
                )

        for parameter in (list(self.input.parameters()) + list(self.l1.parameters()) + list(self.l2.parameters())):
            parameter.requires_grad_(True)

        torch.save(self.state_dict(), save_path)
        print("Training complete. Weights saved.")

    def _policyTensors(self, samples, split_name):
        inputs = torch.tensor(
            [sample[0] for sample in samples],
            dtype=torch.float32,
        )
        targets = torch.zeros(
            (len(samples), self.p2.out_features),
            dtype=torch.float32,
        )
        legal_mask = torch.zeros_like(targets, dtype=torch.bool)
        for row, (_, distribution) in enumerate(samples):
            if not isinstance(distribution, dict) or not distribution:
                raise ValueError(
                    f"{split_name} policy target {row} must be a non-empty action map"
                )
            for action, probability in distribution.items():
                if not isinstance(action, int) or not 0 <= action < self.p2.out_features:
                    raise ValueError(
                        f"{split_name} policy action index is out of range: {action}"
                    )
                if not isinstance(probability, (int, float)) or not torch.isfinite(
                    torch.tensor(probability)
                ).item() or probability < 0:
                    raise ValueError(
                        f"{split_name} policy target has invalid probability: "
                        f"{probability}"
                    )
                targets[row, action] = probability
                legal_mask[row, action] = True
        target_mass = targets.sum(dim=1)
        if not torch.allclose(
            target_mass,
            torch.ones_like(target_mass),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                f"{split_name} policy distributions must sum to 1"
            )
        return inputs, targets, legal_mask

    @staticmethod
    def _softPolicyLoss(logits, targets, legal_mask):
        masked_logits = logits.masked_fill(
            ~legal_mask,
            torch.finfo(logits.dtype).min,
        )
        return -(
            targets * F.log_softmax(masked_logits, dim=1)
        ).sum(dim=1).mean()

    def loadModel(self, path="arcturusv2_weights.pt"):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        print("Loading progress...")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "stateDict" in state:
            state = state["stateDict"]
        incompatible = self.load_state_dict(state, strict=False)
        policyKeys = {"p1.weight", "p1.bias", "p2.weight", "p2.bias"}
        if incompatible.unexpected_keys or set(incompatible.missing_keys) - policyKeys:
            raise RuntimeError(
                "Checkpoint is incompatible with this network: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            print("Value weights restored; policy head initialized randomly.")
        else:
            print("Weights restored.")

def fromExport(line):
    parts = line.strip().split()
    if len(parts) != 19:
        raise ValueError(
            f"Expected 19 exported fields, received {len(parts)}"
        )

    expected_lengths = (144, 144, 132, 132, 132, 132)
    feature_planes = [
        [[0.0 for _ in range(12)] for _ in range(12)]
        for _ in range(14)
    ]

    for field, expected in zip(parts[:6], expected_lengths):
        if len(field) != expected or any(bit not in "01" for bit in field):
            raise ValueError(
                f"Expected a {expected}-bit board field"
            )

    if parts[6] not in ("0", "1"):
        raise ValueError("Turn field must be 0 or 1")

    bit_fields = [[int(bit) for bit in field[::-1]] for field in parts[:6]]
    stone_set, stone_owner, h_stick_set, h_stick_owner, v_stick_set, v_stick_owner = bit_fields
    try:
        scalars = [int(value) for value in parts[7:13]]
        flipped_indices = [int(value) for value in parts[13:19]]
    except ValueError as error:
        raise ValueError("Scalar and flip-history fields must be integers") from error

    if any(value < 0 for value in scalars):
        raise ValueError("Scalar fields must be non-negative")
    if any(index < -1 or index >= 144 for index in flipped_indices):
        raise ValueError("Recent flip indices must be -1 or between 0 and 143")

    for index in range(144):
        row, col = divmod(index, 12)
        feature_planes[0][row][col] = stone_set[index]
        feature_planes[1][row][col] = stone_owner[index]

    for index in range(132):
        row, col = divmod(index, 11)
        feature_planes[2][row][col] = h_stick_set[index]
        feature_planes[3][row][col] = h_stick_owner[index]

        col, row = divmod(index, 11)
        feature_planes[4][row][col] = v_stick_set[index]
        feature_planes[5][row][col] = v_stick_owner[index]

    is_player_one_turn = parts[6] == "1"
    turn_value = 1.0 if is_player_one_turn else 0.0
    feature_planes[6] = [
        [turn_value for _ in range(12)] for _ in range(12)
    ]
    for channel, value, scale in zip(
        range(7, 13),
        scalars,
        (10.0, 10.0, 30.0, 10.0, 10.0, 30.0),
    ):
        normalized_value = value / scale
        feature_planes[channel] = [
            [normalized_value for _ in range(12)] for _ in range(12)
        ]

    for age, index in enumerate(flipped_indices):
        if index >= 0:
            row, col = divmod(index, 12)
            feature_planes[13][row][col] = (age + 1) / 6.0

    return canonicalizeFeatures(feature_planes), is_player_one_turn

def canonicalizeFeatures(features):
    """Express ownership and player resources relative to the side to move."""
    is_player_one_turn = features[6][0][0] > 0.5

    for row in range(12):
        for col in range(12):
            occupied = features[0][row][col]
            stone_is_player_one = features[1][row][col]
            h_stick_exists = features[2][row][col]
            h_stick_is_player_one = features[3][row][col]
            v_stick_exists = features[4][row][col]
            v_stick_is_player_one = features[5][row][col]

            features[1][row][col] = occupied * (
                stone_is_player_one
                if is_player_one_turn
                else 1.0 - stone_is_player_one
            )
            features[3][row][col] = h_stick_exists * (
                h_stick_is_player_one
                if is_player_one_turn
                else 1.0 - h_stick_is_player_one
            )
            features[5][row][col] = v_stick_exists * (
                v_stick_is_player_one
                if is_player_one_turn
                else 1.0 - v_stick_is_player_one
            )
            features[6][row][col] = 1.0

    if not is_player_one_turn:
        for first, second in ((7, 10), (8, 11), (9, 12)):
            features[first], features[second] = (
                features[second],
                features[first],
            )

    return features


    