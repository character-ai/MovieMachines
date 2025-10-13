import os
import logging
import gc

# Third-party library imports
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from optimum.quanto import freeze, qint8, quantize
import torch
from tqdm import tqdm

# Local project imports
from ovi.model_manager import ModelManager
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ovi.utils.model_loading_utils import (init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae,
                                             init_wan_vae_2_2, load_fusion_checkpoint)
from ovi.utils.processing_utils import (preprocess_image_tensor, snap_hw_to_multiple_of_32)


DEFAULT_CONFIG = OmegaConf.load('ovi/configs/inference/inference_fusion.yaml')

class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        print("\n--- Ovi Generation Engine Initialized (lightweight) ---")
        self.device = device
        self.target_dtype = target_dtype
        self.manager = ModelManager(self.device)
        self.config = config
        self.text_model_wrapper = None
        self.audio_latent_channel = None
        self.video_latent_channel = None
        
        # New state variable to track if models are in memory
        self.models_are_loaded = False
        
        print("✅ Engine is ready. Models will be loaded into RAM on first generation.")

    def _load_models(self):
        """Loads all models from disk into RAM and registers them."""
        # If models are already loaded, do nothing.
        if self.models_are_loaded:
            return

        print("\n--- Loading all models into system RAM. This may take a moment... ---")
        config = self.config
        device = "cpu" # Always load to CPU first
        target_dtype = self.target_dtype

        # (This is the heavy loading code moved from the old __init__)
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device, dtype=target_dtype)
        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        text_model_wrapper = init_text_model(config.ckpt_dir, rank=device, cpu_offload=True)
        
        meta_init = True
        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)

        fp8 = config.get("fp8", False)
        checkpoint_path = os.path.join(
            config.ckpt_dir, "Ovi", "model.safetensors" if not fp8 else "model_fp8_e4m3fn.safetensors",
        )
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")

        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8: model = model.to(dtype=target_dtype)
            model = model.to(device="cpu").eval()
            model.set_rope_params()

        int8 = config.get("qint8", False)
        if int8:
            print("› Applying QINT8 quantization to the fusion model...")
            quantize(model, qint8)
            freeze(model)
        
        self.manager.add("fusion", model)
        self.manager.add("vae_video", vae_model_video)
        self.manager.add("vae_audio", vae_model_audio)
        self.manager.add("text", text_model_wrapper.model)

        # Store necessary references
        self.text_model_wrapper = text_model_wrapper
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        
        self.models_are_loaded = True
        print("--- All models successfully loaded into RAM. ---")

    def unload_models(self):
        """Completely removes all models from memory (CPU and GPU)."""
        import psutil
        import ctypes
        
        if not self.models_are_loaded:
            print("[MEM] Models are already unloaded.")
            return

        # Get initial memory usage
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
        print(f"[MEM] Current RAM usage: {mem_before_mb:.1f} MB")
        print("[MEM] Unloading all models from RAM...")
        
        # First, offload everything from GPU to CPU
        self.manager.offload_all_models()
        
        # Clear all references in the manager
        self.manager.clear_all_references()
        
        # Clear engine's own references
        self.text_model_wrapper = None
        self.audio_latent_channel = None
        self.video_latent_channel = None
        
        # More aggressive garbage collection
        print("[MEM] Running garbage collection (this may take a moment)...")
        for _ in range(3):  # Multiple passes help with complex object graphs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Force PyTorch to release memory
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()  # Clean up CUDA IPC resources
        
        # Try to force Python to release memory to OS (platform-specific)
        try:
            if hasattr(ctypes, 'windll'):  # Windows
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            elif hasattr(ctypes, 'CDLL'):  # Linux
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass
        except Exception as e:
            print(f"[MEM] Note: Could not force OS memory release: {e}")
        
        # Final garbage collection
        gc.collect()
        
        # Check memory after cleanup (wait a moment for OS to process)
        import time
        time.sleep(0.5)
        mem_after_mb = process.memory_info().rss / (1024 * 1024)
        freed_mb = mem_before_mb - mem_after_mb
        print(f"[MEM] RAM usage after cleanup: {mem_after_mb:.1f} MB")
        if freed_mb > 10000:
            print(f"[MEM] ✅ Freed approximately {freed_mb:.1f} MB of RAM")
        else:
            print(f"[MEM] ⚠️ ¸ Only {freed_mb:.1f} MB freed immediately. OS will reclaim more memory gradually.")
            print(f"[MEM] This is normal - Python holds memory in reserve for performance.")
        
        self.models_are_loaded = False
        print("[MEM] Models unloaded. Memory cleanup complete.")
    
    @torch.inference_mode()
    def generate(self,
                    text_prompt,
                    image_path=None,
                    video_frame_height_width=None,
                    resolution_budget="720",
                    seed=100,
                    use_tiled_vae=True,                    
                    vae_tile_size=32,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt="",
                    video_latent_length=31,
                    audio_latent_length=157
                ):
        
        params = {
            "Text Prompt": text_prompt, "Image Path": image_path if image_path else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width, "Resolution Budget": resolution_budget,
            "Seed": seed, "VAE Tile Size": vae_tile_size, "Solver": solver_name, "Sample Steps": sample_steps, "Shift": shift,
            "Video Guidance Scale": video_guidance_scale, "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer, "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt, "Video Latent Length": video_latent_length,
            "Audio Latent Length": audio_latent_length,
        }
        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        print("\n========== Generation Parameters ==========")
        print(f"{pretty}")
        print("==========================================")
        
        self._load_models()
        
        try:
            # === Stage 1: Text Encoding ===
            print("› Encoding text prompt...")
            text_model_gpu = self.manager.get('text')
            self.text_model_wrapper.model = text_model_gpu
            self.text_model_wrapper.device = self.device

            text_embeddings = self.text_model_wrapper([text_prompt, video_negative_prompt, audio_negative_prompt], self.device)
            text_embeddings = [emb.to(self.target_dtype).cpu() for emb in text_embeddings] # Offload to CPU
            self.manager.clear_gpu()
            self.text_model_wrapper.model = None
            
            text_embeddings_audio_pos, text_embeddings_video_pos = text_embeddings[0], text_embeddings[0]
            text_embeddings_video_neg, text_embeddings_audio_neg = text_embeddings[1], text_embeddings[2]
            del text_embeddings

            # === Stage 2 (Optional): Image Encoding (for I2V) ===
            is_i2v = image_path is not None
            if resolution_budget == "960": target_area = 960 * 960
            elif resolution_budget == "832": target_area = 832 * 832
            else: target_area = 720 * 720
            
            latents_images = None
            if is_i2v:
                print("› Encoding initial image...")
                vae_video_gpu_wrapper = self.manager.get('vae_video')

                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype, resize_total_area=target_area)
                with torch.no_grad():
                    latents_images = vae_video_gpu_wrapper.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
            else:
                video_h, video_w = video_frame_height_width
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area=target_area)
                video_latent_h, video_latent_w = video_h // 16, video_w // 16

            # === Stage 3: Denoising Loop (Main Model) ===
            print("› Loading main generation model...")
            fusion_model_gpu = self.manager.get('fusion')

            # Move embeddings to GPU for the loop
            text_embeddings_audio_pos = text_embeddings_audio_pos.to(self.device)
            text_embeddings_video_pos = text_embeddings_video_pos.to(self.device)
            text_embeddings_video_neg = text_embeddings_video_neg.to(self.device)
            text_embeddings_audio_neg = text_embeddings_audio_neg.to(self.device)

            scheduler_video, timesteps_video = self.get_scheduler_time_steps(sample_steps, solver_name, self.device, shift)
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(sample_steps, solver_name, self.device, shift)
            
            video_noise = torch.randn((self.video_latent_channel, video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))
            audio_noise = torch.randn((audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))
            
            if is_i2v: latents_images = latents_images.to(self.device, dtype=self.target_dtype)

            _patch_size_h, _patch_size_w = fusion_model_gpu.video_model.patch_size[1], fusion_model_gpu.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h * _patch_size_w)
            max_seq_len_audio = audio_noise.shape[0]

            print("› Generating video and audio frames...")
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    timestep_input = torch.full((1,), t_v, device=self.device)
                    if is_i2v: video_noise[:, :1] = latents_images
                    pos_args = {'audio_context': [text_embeddings_audio_pos], 'vid_context': [text_embeddings_video_pos], 'vid_seq_len': max_seq_len_video, 'audio_seq_len': max_seq_len_audio, 'first_frame_is_clean': is_i2v}
                    pred_vid_pos, pred_audio_pos = fusion_model_gpu(vid=[video_noise], audio=[audio_noise], t=timestep_input, **pos_args)
                    neg_args = {'audio_context': [text_embeddings_audio_neg], 'vid_context': [text_embeddings_video_neg], 'vid_seq_len': max_seq_len_video, 'audio_seq_len': max_seq_len_audio, 'first_frame_is_clean': is_i2v, 'slg_layer': slg_layer}
                    pred_vid_neg, pred_audio_neg = fusion_model_gpu(vid=[video_noise], audio=[audio_noise], t=timestep_input, **neg_args)
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                    video_noise = scheduler_video.step(pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
                    audio_noise = scheduler_audio.step(pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)

            # Offload the main fusion model to make space for the VAEs
            self.manager.clear_gpu()
            
            # === Stage 4: VAE Decoding (Audio) ===
            print("› Decoding audio...")
            vae_audio_gpu = self.manager.get('vae_audio')
            audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)
            generated_audio = vae_audio_gpu.wrapped_decode(audio_latents_for_vae).squeeze().cpu().float().numpy()
            self.manager.clear_gpu()
            
            # === Stage 5: VAE Decoding (Video) ===
            if use_tiled_vae:
                print(f"› Decoding video frames (using {vae_tile_size}x{vae_tile_size} tiles)...")
            else:
                print("› Decoding video frames (Standard, high VRAM)...")

            vae_video_gpu_wrapper = self.manager.get('vae_video')
          
            if is_i2v:
                video_noise[:, :1] = latents_images
           
            video_latents_for_vae = video_noise.unsqueeze(0)

            # --- THE CONDITIONAL LOGIC ---
            if use_tiled_vae:
                generated_video_tensor = vae_video_gpu_wrapper.wrapped_decode_tiled(
                    video_latents_for_vae,
                    tile_x=vae_tile_size,
                    tile_y=vae_tile_size,
                    overlap_x=8,
                    overlap_y=8
                )
            else:
                # Call the standard, non-tiled decode method
                generated_video_tensor = vae_video_gpu_wrapper.wrapped_decode(
                    video_latents_for_vae
                )

            if generated_video_tensor is None:
                # Updated error message for clarity
                decode_method = "Tiled" if use_tiled_vae else "Standard"
                raise RuntimeError(f"{decode_method} VAE decoding failed and returned None.")

            generated_video = generated_video_tensor.squeeze(0).cpu().float().numpy()
            
            print("✅ Generation complete.")
            return generated_video, generated_audio, None

        except Exception as e:
            logging.error(traceback.format_exc())
            return None, None, None
        finally:
            # Ensure GPU is clear at the end of the process
            self.manager.clear_gpu()

    # get_scheduler_time_steps remains unchanged
    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)
        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler,device=device,sigmas=sampling_sigmas)
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, sampling_steps = retrieve_timesteps(sample_scheduler,sampling_steps,device=device,)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler, timesteps