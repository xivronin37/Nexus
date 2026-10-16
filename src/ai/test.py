import torch
import network as nt
from network import fromExport
from learn import train, exportCheckpoint
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tools import constantPredictor, evalModelPerformance

net = nt.Network()
script_dir = os.path.dirname(os.path.abspath(__file__))

INITIAL_WEIGHTS_PATH = os.path.abspath(
    os.path.join(script_dir, "..", "..", "arcturus_weights_gen3n.pt")
)

EVAL_PATH = os.path.abspath(os.path.join(script_dir, "eval_comparison_14ch.pt"))

MODEL_PATH = os.path.abspath(os.path.join(script_dir, "..", "..", "arcturus_weights_gen3n.onnx"))

PRIORS_PATH = os.path.join(script_dir, "priorsData_gen3.pt")

DEST_PATH = os.path.join(script_dir, "compactPriorsData_gen3.pt")



checkpoints = [1]

for num in checkpoints:
    ptPath = os.path.abspath(os.path.join(script_dir, "checkpoints", "gen4", f"e{num}.pt"))
    onnxPath = os.path.abspath(os.path.join(script_dir, "..", "..", f"gen4_4_e{num}.onnx"))
    exportCheckpoint(ptPath, onnxPath)










    


