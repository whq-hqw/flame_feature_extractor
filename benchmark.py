import time
import cv2
import numpy as np

from PIL import Image
import requests
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

from flame_feature_extractor.feature_extractor import PreProcessBatchFace, FeatureExtractorFLAME
from flame_feature_extractor.feature_extractor.preprocess import PreProcessMediaPipe
from flame_feature_extractor.renderer import FlameRenderer

transform = transforms.Compose([
    transforms.PILToTensor()])


def resize_with_smallest_side(image: Image.Image, target_size: int = 256) -> Image.Image:
    # Get the original size
    original_width, original_height = image.size

    # Calculate the scaling factor based on the smallest side
    if original_width < original_height:
        scale_factor = target_size / original_width
    else:
        scale_factor = target_size / original_height

    # Calculate the new size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image


def load_image_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.convert('RGB')
    image = resize_with_smallest_side(image, 256)
    image = transform(image)
    return image


print('Loading image...')
img = load_image_from_url('https://images.pexels.com/photos/1102341/pexels-photo-1102341.jpeg')
imgs = torch.stack([img] * 120)

print('Loading models...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preprocessor = PreProcessBatchFace()
preprocessor = PreProcessMediaPipe()
feature_extractor = FeatureExtractorFLAME().to(device).eval()

renderer = FlameRenderer(
    max_batch_size=128,
    fixed_transform=False,
    n_shape=100,
    n_exp=50,
    scale=5.0,
    device=device.type
)

iters = 10
print('Benchmarking preprocessing...')
st = time.time()
for _ in tqdm(range(iters)):
    preprocessor_output = preprocessor(imgs)

fps = 120 * iters / (time.time() - st)
print('Preprocessing time (CPU) (120 frames):', (time.time() - st) / 50)
print(f"FPS: {fps}")

preprocessor_output = {k: v.to(device) for k, v in preprocessor_output.items()}

print('Benchmarking feature extraction...')
torch.cuda.synchronize()
st = time.time()
for _ in tqdm(range(iters)):
    output = feature_extractor(**preprocessor_output)
torch.cuda.synchronize()
fps = 120 * iters / (time.time() - st)
print(f'Feature extraction time ({device}) (120 frames):', (time.time() - st) / 50)
print(f"FPS: {fps}")

torch.cuda.synchronize()
st = time.time()
out_ims, _ = renderer.render_batch(**output)
torch.cuda.synchronize()
print(f'Rendering time ({device}):', time.time() - st)

# Save the first rendered image as an example, rendering is done in batch
cv2.imwrite('original.png', img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
cv2.imwrite('rendered.png', out_ims.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0])
