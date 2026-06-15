import os
import shutil

import numpy as np
import torch
from PIL import Image

import folder_paths


class SpillImageBatchToDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "directory": (
                    "STRING",
                    {
                        "default": "video/LTXDirector_v1_2_spill_frames",
                        "multiline": False,
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "frame",
                        "multiline": False,
                    },
                ),
                "keep_tail": ("INT", {"default": 12, "min": 0, "max": 4096, "step": 1}),
                "reset_directory": ("BOOLEAN", {"default": False}),
                "use_blended_images": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "blended_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("tail_images", "directory_path", "total_frame_count", "committed_frame_count")
    FUNCTION = "spill"
    CATEGORY = "LTXDirector/memory"

    @staticmethod
    def _resolve_directory(directory: str) -> str:
        directory = (directory or "").strip()
        if not directory:
            directory = "video/LTXDirector_v1_2_spill_frames"
        if os.path.isabs(directory):
            return directory
        return os.path.join(folder_paths.get_output_directory(), directory)

    @staticmethod
    def _count_existing_frames(directory_path: str, filename_prefix: str) -> int:
        if not os.path.isdir(directory_path):
            return 0
        prefix = f"{filename_prefix}_"
        count = 0
        for name in os.listdir(directory_path):
            if name.startswith(prefix) and name.endswith(".png"):
                count += 1
        return count

    @staticmethod
    def _clear_directory(directory_path: str):
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)

    @staticmethod
    def _save_frame(frame: torch.Tensor, file_path: str):
        image = frame.detach().cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(file_path, compress_level=4)

    @staticmethod
    def _select_images(images, blended_images, use_blended_images):
        if use_blended_images and blended_images is not None:
            return blended_images
        return images

    def spill(
        self,
        images,
        directory,
        filename_prefix,
        keep_tail,
        reset_directory,
        use_blended_images,
        blended_images=None,
    ):
        if images is None:
            raise ValueError("images input is required")

        selected_images = self._select_images(images, blended_images, use_blended_images)
        if selected_images is None:
            raise ValueError("selected image batch is required")

        directory_path = self._resolve_directory(directory)
        if reset_directory:
            self._clear_directory(directory_path)
        os.makedirs(directory_path, exist_ok=True)

        image_count = int(selected_images.shape[0])
        keep_tail = max(0, min(int(keep_tail), image_count))
        save_count = image_count - keep_tail

        existing_count = self._count_existing_frames(directory_path, filename_prefix)
        for frame_index in range(save_count):
            file_name = f"{filename_prefix}_{existing_count + frame_index:06d}.png"
            self._save_frame(selected_images[frame_index], os.path.join(directory_path, file_name))

        tail_images = selected_images[save_count:] if keep_tail > 0 else selected_images[:0]
        total_frame_count = existing_count + image_count
        committed_frame_count = existing_count + save_count

        return (tail_images, directory_path, total_frame_count, committed_frame_count)