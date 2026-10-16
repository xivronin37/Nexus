
import torch
import numpy as np


def constantPredictor(trainTargets, valTargets):
    """
    Returns a dict containing cTrain, cVal, oracleMse, fairMse,  distShift,
    totalTrainSamples, and totalValSamples based on provided targets.
    """
    def cleanArray(t):
        if isinstance(t, torch.Tensor):
            # Ensure we are evaluating scalars detached from any active gradient graphs
            npTargets = t.detach().cpu().numpy()
        else:
            npTargets = np.asarray(t)
        return npTargets.reshape(-1).astype(np.float64)
    
    yTrain = cleanArray(trainTargets)
    yVal = cleanArray(valTargets)
    cTrain = np.mean(yTrain)
    cVal = np.mean(yVal)
    varVal = np.var(yVal)
    oracleMse = varVal
    fairMse = varVal + (cVal - cTrain) ** 2
    distShift = abs(cVal - cTrain)
    return {
        "cTrain": float(cTrain),
        "cVal": float(cVal),
        "oracleMse": float(oracleMse),
        "fairMse": float(fairMse),
        "distShift": float(distShift),
        "totalTrainSamples": int(len(yTrain)),
        "totalValSamples": int(len(yVal))
    }

def evalModelPerformance(valMse, baselineMse):
    if baselineMse == 0:
        return {
                "ratio": float("nan"),
                "fractionalImprovement": float("nan"),
                "improvementPercentage": float("nan")
            }

    ratio = float(valMse) / float(baselineMse)
    rSquared = 1.0 - ratio
    
    return {
        "ratio": ratio,                               
        "fractionalImprovement": rSquared,
        "improvementPercentage": rSquared * 100
    }