import time
import cv2
import numpy as np

from PIL import Image
import requests
import torchvision.transforms as transforms
import torch

from flame_feature_extractor.feature_extractor import PreProcessBatchFace, FeatureExtractorFLAME
from flame_feature_extractor.renderer import FlameRenderer

transform = transforms.Compose([
    transforms.PILToTensor()])


def load_image_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image = transform(image)
    return image


img = load_image_from_url('https://github.com/elliottzheng/batch-face/blob/master/examples/obama.jpg?raw=true')

imgs = torch.stack([img] * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocessor = PreProcessBatchFace()
feature_extractor = FeatureExtractorFLAME().to(device).eval()

renderer = FlameRenderer(
    max_batch_size=128,
    fixed_transform=False,
    n_shape=100,
    n_exp=50,
    scale=5.0,
).to(device)

st = time.time()
for _ in range(50):
    preprocessor_output = preprocessor(imgs)

print('Preprocessing time (CPU):', (time.time() - st) / 50)

preprocessor_output = {k: v.to(device) for k, v in preprocessor_output.items()}
torch.cuda.synchronize()
st = time.time()
for _ in range(50):
    output = feature_extractor(mica_images=preprocessor_output['mica_images'],
                               emoca_images=preprocessor_output['emoca_images'])
torch.cuda.synchronize()
print(f'Feature extraction time ({device}):', (time.time() - st) / 50)

# print(output['shape'].shape, output['expression'].shape, output['pose'].shape)

torch.cuda.synchronize()
st = time.time()
out_ims, _ = renderer.render_batch(**output)
torch.cuda.synchronize()
print(f'Rendering time ({device}):', time.time() - st)

# Save the first rendered image as an example, rendering is done in batch
cv2.imwrite('rendered.png', out_ims.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
