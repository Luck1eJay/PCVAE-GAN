import numpy as np
from PIL import Image

def save_phase_image(phase, save_path):
    """
    phase: (H, W) numpy
    """
    phase = phase - phase.min()
    phase = phase / (phase.max() + 1e-8)
    phase = (phase * 255).astype(np.uint8)

    img = Image.fromarray(phase)
    img.save(save_path)
