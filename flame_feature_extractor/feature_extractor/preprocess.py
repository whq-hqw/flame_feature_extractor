import cv2
import torch
import torchvision
from batch_face import RetinaFace

from .mica.insightface import face_align


class PreProcessBatchFace:
    def __init__(self, gpu_id=-1):
        self.detector = RetinaFace(fp16=False, gpu_id=gpu_id)
        self.threshold = 0.95

    def __call__(self, frames):
        """

        :param frames: RGB images of shape (B, 3, H, W), torch tensor
        :return:
        """
        frames = frames.permute(0, 2, 3, 1)  # (B, H, W, 3)

        all_faces = self.detector(frames.cpu().numpy(), threshold=self.threshold)
        frames = frames.permute(0, 3, 1, 2)  # (B, 3, H, W)
        mica_images = []
        emoca_images = []
        for frame, face in zip(frames, all_faces):
            box, kps, score = face[0]  # the first face's detection result

            aimg = face_align.norm_crop(frame.permute(1, 2, 0).cpu().numpy(), landmark=kps)
            blob = cv2.dnn.blobFromImages([aimg], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            mica_image = torch.tensor(blob[0])
            mica_images.append(mica_image)

            size = int((box[2] + box[3] - box[0] - box[1]) / 2 * 1.25)
            center = torch.tensor([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            emoca_image = torchvision.transforms.functional.crop(
                frame.float(),
                top=int(center[1] - size / 2), left=int(center[0] - size / 2),
                height=size, width=size,
            )
            emoca_images.append(emoca_image)

        mica_images = torch.stack(mica_images)
        emoca_images = torch.stack(emoca_images)
        emoca_images = torchvision.transforms.functional.resize(emoca_images, size=224, antialias=True) / 255.0

        return {'mica_images': mica_images, 'emoca_images': emoca_images}
