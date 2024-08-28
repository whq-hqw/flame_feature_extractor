import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from .emoca.emoca_v2 import EMOCAV2
from .mica.mica import MICA


class FeatureExtractorFLAME(nn.Module):
    def __init__(self):
        super().__init__()

        self.emoca_model = EMOCAV2().eval()
        self.mica_model = MICA().eval()

    @torch.no_grad()
    def forward(self, mica_images, emoca_images):
        mica_shape = self.mica_model(mica_images)[:, :100]

        emoca_result = self.emoca_model.encode(emoca_images)

        results = {
            'shape': mica_shape,
            'expression': emoca_result['exp'],
            'pose': emoca_result['pose'],
        }

        return results
