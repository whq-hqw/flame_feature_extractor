#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
import argparse
import pickle

import torch
import torchvision
from tqdm import tqdm

from vqvae.utils.flame_model import FlameRenderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", type=str, default=None)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    params = pickle.load(open(args.file_path, "rb"))
    frame_keys = sorted(list(params.keys()), key=lambda x: int(x.split("_")[-1]))
    flame_renderer = FlameRenderer(fixed_transform=False, device="cuda")
    all_frames = []
    for frame in tqdm(frame_keys[:100]):
        flame_params = params[frame]
        images, _ = flame_renderer.render(**flame_params)
        all_frames.append(images[0].cpu())
    all_frames = torch.stack(all_frames, dim=0).permute(0, 2, 3, 1)
    torchvision.io.write_video("output.mp4", all_frames, 25.0)
