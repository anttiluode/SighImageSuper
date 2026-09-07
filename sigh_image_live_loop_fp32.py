"""Run the Sigh live loop with float32 recursive state on CUDA.

The original live-loop file preserves the first experiment, including its CUDA
float16 state storage. Near 5.9604645e-8 that storage becomes a nonlinear
quantizer and can produce a sparse spectral "last gasp". This launcher keeps the
same GUI/operator but overrides the recursive storage dtype to float32 before the
app is constructed.

Use precision_floor_probe.py for the isolated Q16 control.
"""

import tkinter as tk

import torch

import sigh_image_live_loop as sigh


# Scientific run: do not quantize every recursive generation to float16.
sigh.torch_dtype = torch.float32


if __name__ == "__main__":
    root = tk.Tk()
    app = sigh.LiveCollapseApp(root)
    root.mainloop()
