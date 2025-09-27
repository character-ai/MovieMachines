import torch
import re
import numpy as np
import torch
import cv2
import os
import math
from typing import Tuple
import pandas as pd
from .prompt_reformat import format_prompt

def preprocess_image_tensor(image_path, device, target_dtype, h_w_multiple_of=32, resize_total_area=720*720):
    """Preprocess video data into standardized tensor format and (optionally) resize area."""
    def _parse_area(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]) * int(val[1])
        if isinstance(val, str):
            m = re.match(r"\s*(\d+)\s*[x\*\s]\s*(\d+)\s*$", val, flags=re.IGNORECASE)
            if m:
                return int(m.group(1)) * int(m.group(2))
            if val.strip().isdigit():
                return int(val.strip())
        raise ValueError(f"resize_total_area={val!r} is not a valid area or WxH.")

    def _best_hw_for_area(h, w, area_target, multiple):
        if area_target <= 0:
            return h, w
        ratio_wh = w / float(h)
        area_unit = multiple * multiple
        tgt_units = max(1, area_target // area_unit)
        p0 = max(1, int(round(np.sqrt(tgt_units / max(ratio_wh, 1e-8)))))
        candidates = []
        for dp in range(-3, 4):
            p = max(1, p0 + dp)
            q = max(1, int(round(p * ratio_wh)))
            H = p * multiple
            W = q * multiple
            candidates.append((H, W))
        scale = np.sqrt(area_target / (h * float(w)))
        H_sc = max(multiple, int(round(h * scale / multiple)) * multiple)
        W_sc = max(multiple, int(round(w * scale / multiple)) * multiple)
        candidates.append((H_sc, W_sc))
        def score(HW):
            H, W = HW
            area = H * W
            return (abs(area - area_target), abs((W / max(H, 1e-8)) - ratio_wh))
        H_best, W_best = min(candidates, key=score)
        return H_best, W_best

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image).float().to(device, dtype=target_dtype).unsqueeze(0) ## b c h w
    image_tensor = image_tensor * 2.0 - 1.0 ## -1 to 1

    _, c, h, w = image_tensor.shape
    area_target = _parse_area(resize_total_area)
    if area_target is not None:
        target_h, target_w = _best_hw_for_area(h, w, area_target, h_w_multiple_of)
    else:
        target_h = (h // h_w_multiple_of) * h_w_multiple_of
        target_w = (w // h_w_multiple_of) * h_w_multiple_of

    target_h = max(h_w_multiple_of, int(target_h))
    target_w = max(h_w_multiple_of, int(target_w))

    if (h != target_h) or (w != target_w):
        image_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )

    return image_tensor

def preprocess_audio_tensor(audio, device):
    """Preprocess audio data into standardized tensor format."""
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().squeeze().unsqueeze(0).to(device)
    else:
        audio_tensor = audio.squeeze().unsqueeze(0).to(device)
    return audio_tensor


def calc_dims_from_area(
    aspect_ratio: str,
    total_area: int = 720*720,
    divisible_by: int = 32
) -> Tuple[int, int]:
    """
    Calculate width and height given an aspect ratio (h:w), total area, 
    and divisibility constraint.
    
    Args:
        aspect_ratio (str): Aspect ratio string in format "h:w" (e.g., "9:16").
        total_area (int): Target maximum area (width * height ≤ total_area).
        divisible_by (int): Force width and height to be divisible by this value.
    
    Returns:
        (width, height): Tuple of integers that satisfy constraints.
    """
    # Parse aspect ratio string
    h_ratio, w_ratio = map(int, aspect_ratio.split(":"))

    # Reduce ratio
    gcd = math.gcd(h_ratio, w_ratio)
    h_ratio //= gcd
    w_ratio //= gcd

    # Scaling factor
    k = math.sqrt(total_area / (h_ratio * w_ratio))

    # Floor to multiples of divisible_by
    height = (int(k * h_ratio) // divisible_by) * divisible_by
    width  = (int(k * w_ratio) // divisible_by) * divisible_by

    # Safety check: avoid 0
    height = max(height, divisible_by)
    width  = max(width, divisible_by)

    return height, width



def validate_and_process_user_prompt(text_prompt: str, image_path: str = None) -> str:
    if not isinstance(text_prompt, str):
        raise ValueError("User input must be a string")

    # Normalize whitespace
    text_prompt = text_prompt.strip()

    # Check if it's a file path that exists
    if os.path.isfile(text_prompt):
        _, ext = os.path.splitext(text_prompt.lower())
        
        if ext == ".csv":
            df = pd.read_csv(text_prompt)
        elif ext == ".tsv":
            df = pd.read_csv(text_prompt, sep="\t")
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .csv and .tsv are allowed.")

        assert "text_prompt" in df.keys() and "image_path" in df.keys(), f"Missing required columns in TSV file."
        text_prompts = list(map(lambda t: format_prompt(str(t)), df["text_prompt"]))
        image_paths = list(df["image_path"])

        assert all(os.path.isfile(p) for p in image_paths), "One or more image paths in the TSV file do not exist."
    
    else:
        assert image_path is None or os.path.isfile(image_path), f"Image path is not None but {image_path} does not exist."
        text_prompt = format_prompt(text_prompt)
        text_prompts = [text_prompt]
        image_paths = [image_path]

    return text_prompts, image_paths


def format_prompt_for_filename(text: str) -> str:
    # remove anything inside <...>
    no_tags = re.sub(r"<.*?>", "", text)
    # replace spaces and slashes with underscores
    safe = no_tags.replace(" ", "_").replace("/", "_")
    # truncate to 50 chars
    return safe[:50]