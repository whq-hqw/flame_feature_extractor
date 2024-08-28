# flame_feature_extractor

## Installation

```bash
pip install -r requirements_pre.txt
pip install -r requirements_post.txt
bash build_resources.sh
```

## Usage

```python
import cv2
import numpy as np
from flame_feature_extractor.feature_extractor import PreProcessBatchFace, FeatureExtractorFLAME
from flame_feature_extractor.renderer import FlameRenderer


imgs = ...   # batch of RGB images of shape (B, 3, H, W) , where B is the batch size or could be frames of a video

preprocessor = PreProcessBatchFace()
feature_extractor = FeatureExtractorFLAME()
renderer = FlameRenderer(
    max_batch_size=128,
    fixed_transform=False,
    n_shape=100,
    n_exp=50,
    scale=5.0,
)
preprocessor_output = preprocessor(imgs)
output = feature_extractor(mica_images=preprocessor_output['mica_images'], 
                           emoca_images=preprocessor_output['emoca_images'])

print(output['shape'].shape, output['expression'].shape, output['pose'].shape)

out_ims, _ = renderer.render_batch(**output)

# Save the first rendered image as an example, rendering is done in batch
cv2.imwrite('rendered.png', out_ims.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
```
