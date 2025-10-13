import os
import torch
import gc
import devicetorch
import warnings
import traceback

from pathlib import Path
from huggingface_hub import snapshot_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

# Conditional import for GFPGAN
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_ESRGAN_PATH = _MODULE_DIR / "model_esrgan"
MODEL_GFPGAN_PATH = _MODULE_DIR / "model_gfpgan"

class ESRGANUpscaler:
    def __init__(self, device: torch.device):
        self.device = device
        self.model_dir = Path(MODEL_ESRGAN_PATH)
        self.gfpgan_model_dir = Path(MODEL_GFPGAN_PATH)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.gfpgan_model_dir, exist_ok=True)

        self.supported_models = {
            "RealESRGAN_x2plus": {
                "filename": "RealESRGAN_x2plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "hf_repo_id": None, "scale": 2, "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                "description": "General purpose. Faster than x4 models due to smaller native output. Best choice for Ovi."
            },
            "RealESRGAN_x4plus": {
                "filename": "RealESRGAN_x4plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "hf_repo_id": None, "scale": 4, "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "description": "General purpose. Prioritizes sharpness & detail. Good for most videos."
            },
            "RealESRNet_x4plus": {
                "filename": "RealESRNet_x4plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                "hf_repo_id": None, "scale": 4, "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "description": "Similar to RealESRGAN_x4plus, but trained for higher fidelity, often yielding smoother results."
            },
            "RealESR-general-x4v3": {
                "filename": "realesr-general-x4v3.pth", "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                "wdn_filename": "realesr-general-wdn-x4v3.pth", "wdn_file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "scale": 4, "model_class": SRVGGNetCompact,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
                "description": "Versatile SRVGG-based. Balances detail & naturalness. Has adjustable denoise strength."
            },
            "RealESRGAN_x4plus_anime_6B": {
                "filename": "RealESRGAN_x4plus_anime_6B.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "hf_repo_id": None, "scale": 4, "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                "description": "Optimized for anime. Lighter 6-block version of x4plus for faster anime upscaling."
            },
            "RealESR_AnimeVideo_v3": {
                "filename": "realesr-animevideov3.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                "hf_repo_id": None, "scale": 4, "model_class": SRVGGNetCompact,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
                "description": "Specialized SRVGG-based model for anime. Often excels with animated content."
            }
        }

        self.upsamplers: dict[str, dict[str, Any]] = {}
        self.face_enhancer: GFPGANer | None = None

    def _ensure_model_downloaded(self, model_key: str, target_dir: Path | None = None, is_gfpgan: bool = False, is_wdn_companion: bool = False) -> bool:
        current_model_dir = target_dir or (self.gfpgan_model_dir if is_gfpgan else self.model_dir)
        model_info_source = {}
        actual_model_filename = ""

        if is_gfpgan:
            model_info_source = {"filename": "GFPGANv1.4.pth", "file_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"}
            actual_model_filename = model_info_source["filename"]
        else:
            if model_key not in self.supported_models:
                print(f"ERROR: ESRGAN model key '{model_key}' not supported.")
                return False
            
            model_details = self.supported_models[model_key]
            if is_wdn_companion:
                if "wdn_filename" not in model_details or "wdn_file_url" not in model_details:
                    print(f"ERROR: WDN companion model details missing for '{model_key}'.")
                    return False
                model_info_source = {"filename": model_details["wdn_filename"], "file_url": model_details["wdn_file_url"]}
                actual_model_filename = model_details["wdn_filename"]
            else:
                model_info_source = model_details
                actual_model_filename = model_details["filename"]

        model_path = current_model_dir / actual_model_filename

        if not model_path.exists():
            log_prefix = "WDN " if is_wdn_companion else ""
            print(f"INFO: {log_prefix}Model '{actual_model_filename}' not found. Downloading...")
            try:
                url = model_info_source.get("file_url")
                if not url:
                    print(f"ERROR: No download URL specified for '{actual_model_filename}'.")
                    return False
                
                print(f"INFO: Attempting download from URL: {url}")
                load_file_from_url(url=url, model_dir=str(current_model_dir), progress=True, file_name=actual_model_filename)
                
                if model_path.exists():
                    print(f"SUCCESS: {log_prefix}Model '{actual_model_filename}' downloaded.")
                else:
                    print(f"ERROR: Download failed for '{actual_model_filename}'. File not found after download attempt.")
                    return False
            except Exception as e:
                print(f"ERROR: Failed to download {log_prefix}model '{actual_model_filename}': {e}")
                traceback.print_exc()
                return False
        return True

    def load_model(self, model_key: str, tile_size: int = 0, denoise_strength: float | None = None) -> RealESRGANer | None:
        if model_key not in self.supported_models:
            print(f"ERROR: ESRGAN model key '{model_key}' not supported.")
            return None

        current_config_signature = (tile_size, denoise_strength if "v3" in model_key else None)
        
        if model_key in self.upsamplers:
            existing_config = self.upsamplers[model_key]
            existing_config_signature = (existing_config.get('tile_size', 0), existing_config.get('denoise_strength') if "v3" in model_key else None)

            if existing_config.get("upsampler") and existing_config_signature == current_config_signature:
                print(f"INFO: ESRGAN model '{model_key}' with the same configuration is already loaded.")
                return existing_config["upsampler"]
            elif existing_config.get("upsampler"):
                print(f"INFO: ESRGAN model '{model_key}' config changed. Reloading with new settings.")
                self.unload_model(model_key)

        if not self._ensure_model_downloaded(model_key): return None

        model_info = self.supported_models[model_key]
        model_path = str(self.model_dir / model_info["filename"])
        dni_weight = None
        
        print(f"INFO: Loading ESRGAN model '{model_info['filename']}' (Scale: {model_info['scale']}x, Tile: {tile_size or 'Auto'})...")

        if "v3" in model_key and denoise_strength is not None and 0.0 <= denoise_strength < 1.0:
            if not self._ensure_model_downloaded(model_key, is_wdn_companion=True): return None
            wdn_model_path = str(self.model_dir / model_info["wdn_filename"])
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1.0 - denoise_strength]
            print(f"INFO: Applying denoise strength (DNI): {denoise_strength:.2f}")

        try:
            model_arch = model_info["model_class"](**model_info["model_params"])
            gpu_id = self.device.index if self.device.type == 'cuda' else None
            use_half = self.device.type == 'cuda'
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")
                
                upsampler = RealESRGANer(
                    scale=model_info["scale"], model_path=model_path, dni_weight=dni_weight, model=model_arch,
                    tile=tile_size, tile_pad=10, pre_pad=0, half=use_half, gpu_id=gpu_id
                )
            
            self.upsamplers[model_key] = {
                "upsampler": upsampler, "tile_size": tile_size, "native_scale": model_info["scale"],
                "denoise_strength": denoise_strength if "v3" in model_key else None
            }
            print(f"SUCCESS: ESRGAN model '{model_info['filename']}' loaded.")
            return upsampler
        except Exception as e:
            print(f"ERROR: Failed to load ESRGAN model '{model_info['filename']}': {e}")
            traceback.print_exc()
            if model_key in self.upsamplers: del self.upsamplers[model_key]
            return None

    def _load_face_enhancer(self, model_name="GFPGANv1.4.pth", bg_upsampler=None) -> bool:
        if not GFPGAN_AVAILABLE:
            print("WARNING: GFPGAN library not available. Face enhancement is disabled.")
            return False
        if self.face_enhancer:
            if bg_upsampler and self.face_enhancer.bg_upsampler != bg_upsampler:
                print("INFO: Background upsampler changed. Re-initializing GFPGAN...")
                self._unload_face_enhancer()
            else:
                print("INFO: GFPGAN face enhancer already loaded.")
                return True

        if not self._ensure_model_downloaded(model_key=model_name, is_gfpgan=True): return False

        gfpgan_model_path = str(self.gfpgan_model_dir / model_name)
        print(f"INFO: Loading GFPGAN face enhancer from {gfpgan_model_path}...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.face_enhancer = GFPGANer(
                    model_path=gfpgan_model_path, upscale=1, arch='clean',
                    channel_multiplier=2, bg_upsampler=bg_upsampler, device=self.device
                )
            print("SUCCESS: GFPGAN face enhancer loaded.")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load GFPGAN face enhancer: {e}")
            traceback.print_exc()
            self.face_enhancer = None
            return False

    def _unload_face_enhancer(self):
        if self.face_enhancer:
            print("INFO: Unloading GFPGAN face enhancer...")
            del self.face_enhancer
            self.face_enhancer = None
            gc.collect()
            if self.device.type == 'cuda': torch.cuda.empty_cache()
            print("SUCCESS: GFPGAN face enhancer unloaded.")

    def unload_model(self, model_key: str):
        if model_key in self.upsamplers:
            print(f"INFO: Unloading ESRGAN model '{model_key}'...")
            config = self.upsamplers.pop(model_key)
            upsampler_instance = config.get("upsampler")
            
            if self.face_enhancer and self.face_enhancer.bg_upsampler == upsampler_instance:
                self._unload_face_enhancer()

            del upsampler_instance
            gc.collect(); devicetorch.empty_cache(torch)
            print(f"SUCCESS: ESRGAN model '{model_key}' unloaded and memory cleared.")

    def upscale_frame(self, frame_np_array, model_key: str, target_outscale_factor: float, enhance_face: bool = False):
        """
        Upscales a single frame using the specified model and target output scale.
        """
        config = self.upsamplers.get(model_key)
        upsampler: RealESRGANer | None = None
        current_tile_size = 0
        model_native_scale = 0

        if config and config.get("upsampler"):
            upsampler = config["upsampler"]
            current_tile_size = config.get("tile_size", 0)
            model_native_scale = config.get("native_scale", 0)
            if model_native_scale == 0:
                print(f"ERROR: Native scale for model '{model_key}' is 0 or not found in config.")
                return None
        
        if upsampler is None:
            print(f"WARNING: ESRGAN model '{model_key}' not pre-loaded. Attempting to load now (with default Tile: Auto)...")
            tile_to_load_with = config.get("tile_size", 0) if config else 0
            upsampler = self.load_model(model_key, tile_size=tile_to_load_with)
            if upsampler is None:
                print(f"ERROR: Failed to auto-load ESRGAN model '{model_key}'. Cannot upscale.")
                return None
            
            loaded_config = self.upsamplers.get(model_key)
            if loaded_config:
                current_tile_size = loaded_config.get("tile_size", 0)
                model_native_scale = loaded_config.get("native_scale", 0)
                if model_native_scale == 0:
                    print(f"ERROR: Native scale for auto-loaded model '{model_key}' is 0.")
                    return None
            else:
                print(f"ERROR: Config for auto-loaded model '{model_key}' not found.")
                return None

        if not (0.25 <= target_outscale_factor <= model_native_scale):
            print(
                f"WARNING: Target outscale factor {target_outscale_factor:.2f}x is outside the recommended range "
                f"(0.25x to {model_native_scale:.2f}x) for model '{model_key}'. "
                f"Adjusting to model's native scale of {model_native_scale:.2f}x."
            )
            target_outscale_factor = float(model_native_scale)

        if enhance_face:
            if not self.face_enhancer or (hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler != upsampler):
                print("INFO: Face enhancement requested, loading/re-configuring GFPGAN...")
                self._load_face_enhancer(bg_upsampler=upsampler)
            
            if not self.face_enhancer:
                print("WARNING: GFPGAN could not be loaded. Proceeding without face enhancement.")
                enhance_face = False

        try:
            # Convert RGB frame from imageio to BGR for the models
            img_bgr = frame_np_array[:, :, ::-1]
            
            if enhance_face and self.face_enhancer:
                # First, clean the face with GFPGAN
                _, _, cleaned_img_bgr = self.face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                # Then, upscale the cleaned image with RealESRGAN
                output_bgr, _ = upsampler.enhance(cleaned_img_bgr, outscale=float(target_outscale_factor))
            else:
                # Just upscale with RealESRGAN
                output_bgr, _ = upsampler.enhance(img_bgr, outscale=float(target_outscale_factor))
            
            # Convert the final BGR image back to RGB for video writing
            output_rgb = output_bgr[:, :, ::-1]
            return output_rgb
            
        except Exception as e:
            tile_msg = str(current_tile_size) if current_tile_size > 0 else 'Auto'
            face_msg = " + Face Enhance" if enhance_face else ""
            print(f"ERROR during upscaling (Model: {model_key}{face_msg}, Scale: {target_outscale_factor:.2f}x, Tile: {tile_msg}): {e}")
            
            if "out of memory" in str(e).lower() and self.device.type == 'cuda':
                print("HINT: CUDA Out of Memory. Consider using a smaller tile size or enabling 'Streaming' mode for large videos.")
                devicetorch.empty_cache(torch)
            else:
                traceback.print_exc()
            return None