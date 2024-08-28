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


if __name__ == '__main__':
    fname = 'face4'
    ext = '.jpeg'
    img = read_image(fname + ext, mode=ImageReadMode.RGB)
    imgs = torch.stack([img]*1)

    # preprocessor = PreProcessCombined()
    preprocessor = PreProcessBatchFace()
    feature_extractor = FeatureExtractorFLAME()
    renderer = FlameRenderer(
        max_batch_size=128,
        fixed_transform=False,
        n_shape=100,
        n_exp=50,
        scale=5.0,
    )
    st = time.time()
    preprocessor_output = preprocessor(imgs)
    print('Preprocessing time (CPU):', time.time() - st)

    emoca_image = preprocessor_output['emoca_images'][0] * 255
    mica_image = preprocessor_output['mica_images'][0] * 255
    print(emoca_image.shape, mica_image.shape)
    cv2.imwrite(fname + '_emoca.png', emoca_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    cv2.imwrite(fname + '_mica.png', mica_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    st = time.time()
    output = feature_extractor(**preprocessor_output)
    print('Feature extraction time (CPU):', time.time() - st)

    print(output['shape'].shape, output['expression'].shape, output['pose'].shape)

    out_ims, _ = renderer.render_batch(**output)

    cv2.imwrite(fname + '_mesh.png', out_ims.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
