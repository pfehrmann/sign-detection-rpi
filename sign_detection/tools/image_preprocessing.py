"""
Various methods for image preprocessing
"""

import cv2
import numpy as np


def set_average_value(image, val):
    average = np.average(cv2.mean(image)[:3])
    lut = np.array(range(0, 256)) * (val / average)
    lut = lut.clip(0, 255).astype(np.uint8)
    res = cv2.LUT(src=image.astype(np.uint8), lut=lut)
    return res
