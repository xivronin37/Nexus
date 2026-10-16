import os
import network as nt
from learn import exportOnnx

net = nt.Network()
script_dir = os.path.dirname(os.path.abspath(__file__))

net.loadModel(
    os.path.abspath(
        os.path.join(script_dir, "..", "..", "arcturus_weights_gen2.pt")
    )
)

net.eval()

exportOnnx(
    net,
    filename=os.path.abspath(
        os.path.join(script_dir, "..", "..", "arcturus_weights_gen2_policy.onnx")
    ),
)