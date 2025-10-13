# Standard library imports
import argparse
import asyncio
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import json
from datetime import datetime
from functools import wraps
from pathlib import Path
from io import StringIO

# Third-party library imports
import gradio as gr
import torch
from huggingface_hub import snapshot_download

# Local project imports
from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text, scale_hw_to_area_divisible

from toolbox.toolbox import OviToolboxProcessor
from toolbox.system_monitor import SystemMonitor

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)    
    
# Try to suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    import socket # Required for the ConnectionResetError
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    def silence_connection_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError):
                pass
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    from asyncio import proactor_events
    if hasattr(proactor_events, '_ProactorBasePipeTransport'):
        proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_connection_errors(
            proactor_events._ProactorBasePipeTransport._call_connection_lost
        )
            
# ----------------------------
# Create required directories
# ----------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# ----------------------------
# Parse CLI Args
# ----------------------------
parser = argparse.ArgumentParser(description="Ovi Joint Video + Audio Gradio Demo")
parser.add_argument(
    "--use_image_gen",
    action="store_true",
    help="Enable image generation UI with FluxPipeline"
)
parser.add_argument(
    "--fp8",
    action="store_true",
    help="Enable 8 bit quantization of the fusion model",
)
parser.add_argument(
    "--qint8",
    action="store_true",
    help="Enable 8 bit quantization of the fusion model. No need to download additional models.",
)
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP address, LAN access changed to 0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="Use port")
parser.add_argument("--share", action="store_true", help="Enable gradio sharing")
parser.add_argument("--mcp_server", action="store_true", help="Enable MCP service")
args = parser.parse_args()

ovi_toolbox_processor = OviToolboxProcessor()

# ----------------------------
# Model Download Logic
# ----------------------------
def check_and_download_models():
    """
    Checks for model files and downloads them from Hugging Face if missing.
    """
    base_ckpt_dir = "./ckpts"
    
    models_to_check = [
        {
            "name": "Wan2.2-TI2V-5B",
            "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
            "local_dir": os.path.join(base_ckpt_dir, "Wan2.2-TI2V-5B"),
            "allow_patterns": [
                "google/*",
                "models_t5_umt5-xxl-enc-bf16.pth", 
                "Wan2.2_VAE.pth"
            ],
            "check_files": [
                os.path.join(base_ckpt_dir, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(base_ckpt_dir, "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
            ]
        },
        {
            "name": "MMAudio",
            "repo_id": "hkchengrex/MMAudio",
            "local_dir": os.path.join(base_ckpt_dir, "MMAudio"),
            "allow_patterns": ["ext_weights/best_netG.pt", "ext_weights/v1-16.pth"],
            "check_files": [
                os.path.join(base_ckpt_dir, "MMAudio", "ext_weights", "best_netG.pt"),
                os.path.join(base_ckpt_dir, "MMAudio", "ext_weights", "v1-16.pth")
            ]
        },
    ]
    
    if args.fp8:
        ovi_model_to_check = {
            "name": "Ovi (FP8 Quantized)",
            "repo_id": "rkfg/Ovi-fp8_quantized",
            "local_dir": os.path.join(base_ckpt_dir, "Ovi"),
            "allow_patterns": ["*.safetensors"], 
            "check_files": [os.path.join(base_ckpt_dir, "Ovi", "model_fp8_e4m3fn.safetensors")]
        }
    else:
        ovi_model_to_check = {
            "name": "Ovi (Full Model)",
            "repo_id": "chetwinlow1/Ovi",
            "local_dir": os.path.join(base_ckpt_dir, "Ovi"),
            "allow_patterns": ["model.safetensors"],
            "check_files": [os.path.join(base_ckpt_dir, "Ovi", "model.safetensors")]
        }
    models_to_check.append(ovi_model_to_check)
    
    print()
    print("--- Checking for required model weights ---")
    for model in models_to_check:
        all_files_exist = True
        
        for file_path in model["check_files"]:
            if not os.path.exists(file_path):
                print(f"    ❌ Missing file: {file_path}")
                all_files_exist = False

        if all_files_exist:
            print(f"✅ '{model['name']}' is present.")
            continue

        print(f"⚠️ Weights for '{model['name']}' are incomplete. Starting download...")
        try:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=model["local_dir"],
                local_dir_use_symlinks=False,
                allow_patterns=model["allow_patterns"],
            )
            print(f"✅ Download complete for '{model['name']}'.")
        except Exception as e:
            print(f"❌ Download failed for '{model['name']}'. Please check your network connection.")
            print(f"   Error details: {e}")
    print("--- All model checks complete. ---")
    
check_and_download_models()

# Initialize OviFusionEngine
use_image_gen = args.use_image_gen
fp8 = args.fp8
qint8 = args.qint8
print(f"Starting Gradio UI... {use_image_gen=}, {fp8=}, {qint8=}")

if use_image_gen:
    DEFAULT_CONFIG["mode"] = "t2i2v"
else:
    DEFAULT_CONFIG["mode"] = "t2v"
    
DEFAULT_CONFIG["fp8"] = fp8
DEFAULT_CONFIG["qint8"] = qint8
ovi_engine = OviFusionEngine()
flux_model = None

if use_image_gen:
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
    flux_model.enable_model_cpu_offload()
print("loaded model")

# Apply startup temp clearing based on settings
startup_clear_message = ovi_toolbox_processor.apply_startup_clear_temp()
if startup_clear_message:
    print(startup_clear_message)

# Load saved settings for UI defaults
saved_settings = ovi_toolbox_processor.load_all_settings()

esrgan_model_choices = list(ovi_toolbox_processor.esrgan_upscaler.supported_models.keys())
default_esrgan_model = esrgan_model_choices[0] if esrgan_model_choices else None

# Get the model to use for initial slider values (saved model or default)
model_for_initial_values = saved_settings.get("upscale_model", default_esrgan_model)
initial_model_info_update, initial_slider_update, initial_denoise_update = ovi_toolbox_processor.get_model_info_and_update_scale_slider(model_for_initial_values)

# Use saved settings or defaults for UI components
default_upscale_model = saved_settings.get("upscale_model", default_esrgan_model)
default_upscale_factor = saved_settings.get("upscale_factor", initial_slider_update.get('value', 2.0))
default_upscale_tile_size = saved_settings.get("upscale_tile_size", 0)
default_upscale_enhance_face = saved_settings.get("upscale_enhance_face", False)
default_upscale_streaming = saved_settings.get("upscale_use_streaming", False)
default_denoise_strength = saved_settings.get("denoise_strength", 0.5)
default_fps_mode = saved_settings.get("fps_mode", "2x Frames")
default_speed_factor = saved_settings.get("speed_factor", 1.0)
default_frames_streaming = saved_settings.get("frames_use_streaming", False)
default_export_format = saved_settings.get("export_format", "MP4")
default_export_quality = saved_settings.get("export_quality", 85)
default_export_max_width = saved_settings.get("export_max_width", 1024)
default_export_name = saved_settings.get("export_name", "")

def generate_video(
    text_prompt,
    image,
    video_height,
    video_width,
    video_seed,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    video_negative_prompt,
    audio_negative_prompt,
    autosave_video,
    resolution_budget,
    video_duration,
    use_tiled_vae,
    vae_tile_size,
    progress=gr.Progress()
):
    import sys
    from io import StringIO
    import queue
    import threading
    
    # Create a queue for console updates
    console_queue = queue.Queue()
    output_buffer = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    generation_complete = threading.Event()
    last_line = ""
    
    class TeeOutput:
        def __init__(self, *streams, is_stderr=False):
            self.streams = streams
            self.is_stderr = is_stderr
        
        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()
            if data:
                # Handle carriage returns for tqdm progress bars
                if '\r' in data and self.is_stderr:
                    # This is a progress bar update
                    console_queue.put(("PROGRESS", data))
                elif data.strip():
                    console_queue.put(("TEXT", data))
        
        def flush(self):
            for stream in self.streams:
                stream.flush()
    
    # Redirect both stdout and stderr
    sys.stdout = TeeOutput(original_stdout, output_buffer)
    sys.stderr = TeeOutput(original_stderr, output_buffer, is_stderr=True)
    
    def run_generation():
        try:
            image_path = None
            if image is not None:
                image_path = image

            video_frame_height_width = [video_height, video_width]
            
            # Parse the resolution budget
            if "960" in resolution_budget:
                budget_value = "960"
            elif "832" in resolution_budget:
                budget_value = "832"
            else:
                budget_value = "720"
            
            # Dynamic Latent Length Calculation
            video_latents_per_second = 31 / 5.0
            audio_latents_per_second = 157 / 5.0
            vid_len = int(round(video_duration * video_latents_per_second))
            aud_len = int(round(video_duration * audio_latents_per_second))
            
            generated_video, generated_audio, _ = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_path,
                video_frame_height_width=video_frame_height_width,
                resolution_budget=budget_value,
                seed=video_seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
                use_tiled_vae=use_tiled_vae,
                vae_tile_size=vae_tile_size,            
                video_latent_length=vid_len,
                audio_latent_length=aud_len,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"Ovi_{timestamp}.mp4"
            
            if autosave_video:
                output_path = os.path.join("outputs", output_filename)
            else:
                output_path = os.path.join(tempfile.gettempdir(), output_filename)

            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            console_queue.put(("COMPLETE", output_path))
            
        except Exception as e:
            error_msg = f"Error during video generation: {e}"
            print(error_msg)
            console_queue.put(("ERROR", str(e)))
        finally:
            generation_complete.set()
    
    # Start generation in background thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    
    # Yield updates as they come in
    console_text = "🚀 Starting generation...\n\n"
    yield None, console_text
    
    try:
        while not generation_complete.is_set() or not console_queue.empty():
            try:
                msg = console_queue.get(timeout=0.1)
                msg_type, msg_data = msg if isinstance(msg, tuple) else ("TEXT", msg)
                
                if msg_type == "COMPLETE":
                    console_text = output_buffer.getvalue()
                    yield msg_data, console_text
                    break
                elif msg_type == "ERROR":
                    console_text = output_buffer.getvalue() + f"\n\n❌ Error: {msg_data}"
                    yield None, console_text
                    break
                elif msg_type == "PROGRESS":
                    # Handle tqdm progress bar with carriage return
                    # Split lines and take the last one (most recent progress)
                    lines = console_text.split('\n')
                    progress_text = msg_data.strip('\r\n')
                    
                    if progress_text:
                        # Check if we're updating an existing progress line
                        if lines and 'it [' in lines[-1] and 'it [' in progress_text:
                            # Replace the last line with updated progress
                            lines[-1] = progress_text
                        else:
                            # Add as new line
                            lines.append(progress_text)
                        
                        console_text = '\n'.join(lines)
                        yield None, console_text
                elif msg_type == "TEXT":
                    # Regular text - ensure proper line breaks
                    if not console_text.endswith('\n') and not msg_data.startswith('\n'):
                        console_text += '\n'
                    console_text += msg_data
                    yield None, console_text
                    
            except queue.Empty:
                continue
                
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        gen_thread.join(timeout=1)

def prepare_video_seed(randomize_seed, current_seed):
    if randomize_seed:
        return random.randint(0, 2**32 - 1)
    return current_seed
    
def generate_image(text_prompt, image_seed, randomize_image_seed, image_height, image_width):
    if flux_model is None:
        return None, image_seed
        
    # If randomize seed is checked, generate a random seed
    if randomize_image_seed:
        image_seed = random.randint(0, 2**32 - 1)

    text_prompt = clean_text(text_prompt)
    print(f"Generating image with prompt='{text_prompt}', seed={image_seed}, size=({image_height},{image_width})")

    image_h, image_w = scale_hw_to_area_divisible(image_height, image_width, area=1024 * 1024)
    image = flux_model(
        text_prompt,
        height=image_h,
        width=image_w,
        guidance_scale=4.5,
        generator=torch.Generator().manual_seed(int(image_seed))
    ).images[0]

    # Save the temporary image in the 'tmp' directory
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="tmp")
    image.save(tmpfile.name)
    # Return the path and the updated seed value
    return tmpfile.name, image_seed

def save_video_manually(video_path):
    if video_path and os.path.exists(video_path):
        filename = os.path.basename(video_path)
        destination = os.path.join("outputs", filename)
        shutil.move(video_path, destination)
        return f'<div style="padding: 8px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724;">✅ Video saved to {destination}</div>'
    return '<div style="padding: 8px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404;">⚠️ No video to save or file not found.</div>'

def open_output_folder():
    folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        subprocess.run(["explorer", folder_path])
    elif platform.system() == "Darwin": # macOS
        subprocess.Popen(["open", folder_path])
    else: # Linux
        subprocess.Popen(["xdg-open", folder_path])
    return '<div style="padding: 8px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; color: #0c5460;">📂 Opened output folder.</div>'

def unload_models_from_ram():
    if ovi_engine:
        ovi_engine.unload_models()
        return '<div style="padding: 8px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724;">✅ Models unloaded from RAM. Ready for post-processing or other tasks.</div>'
    return '<div style="padding: 8px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">⚠️ Engine not initialized.</div>'

def handle_start_pipeline(
    active_tab_index, single_video_path, batch_video_paths, selected_ops,
    # Upscale params
    model_key, scale_factor, tile_size, enhance_face, denoise_strength, upscale_use_streaming,
    # Frame Adjust params
    fps_mode, speed_factor, frames_use_streaming,
    # Export params
    export_format, quality, max_width, output_name,
    progress=gr.Progress()
):
    # Determine input paths based on the active tab
    if active_tab_index == 1 and batch_video_paths:
        input_paths = [file.name for file in batch_video_paths]
        if not input_paths:
            return None, "⚠️ Batch Input tab is active, but no files were provided."
    elif active_tab_index == 0 and single_video_path:
        input_paths = [single_video_path]
    else:
        return None, "⚠️ No input video found in the active tab. Please upload a video."

    if not selected_ops:
        return None, "⚠️ No operations selected. Please check at least one box in 'Pipeline Steps'."

    # Pack parameters for the processor
    params = {
        "upscale": {
            "model_key": model_key, "scale_factor": scale_factor, "tile_size": tile_size,
            "enhance_face": enhance_face, "denoise_strength": denoise_strength,
            "use_streaming": upscale_use_streaming
        },
        "frame_adjust": {
            "fps_mode": fps_mode, "speed_factor": speed_factor, "use_streaming": frames_use_streaming
        },
        "export": {
            "export_format": export_format, "quality": quality, "max_width": max_width, "output_name": output_name
        }
    }
    
    if len(input_paths) > 1:
        # Batch processing
        final_video, message = ovi_toolbox_processor.process_batch(input_paths, selected_ops, params, progress)
    else:
        # Single video processing
        temp_video, message = ovi_toolbox_processor.process_pipeline(input_paths[0], selected_ops, params, progress)
        final_video = None
        if temp_video:
            if ovi_toolbox_processor.autosave_enabled:
                temp_path = Path(temp_video)
                final_path = ovi_toolbox_processor.output_dir / temp_path.name
                final_video = ovi_toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
                message += f"\n✅ Autosaved result to: {final_path}"
            else:
                final_video = temp_video # Leave in temp folder for manual save
                message += "\nℹ️ Autosave is off. Result is in a temporary folder. Use 'Manual Save' to keep it."

    return final_video, message


css = """
.video-size video, .video-size img { max-height: 60vh; object-fit: contain; }
#app-video-player .source-selection, #toolbox-video-player .source-selection { display: none !important; }
#start-btn { margin-top: -14px !important; }
"""

footer_html = """
<div style="
    text-align: center; 
    padding: 10px; 
    margin-top: 5px; 
    font-family: sans-serif;
">
    <hr style="border: 0; height: 1px; background: #333; margin-bottom: 10px;">    
    <h2 style="margin-bottom: 5px;"> Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation </h2>
    
    <!-- Container for the badges -->
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; font-size: 0.8em;">
        
        <!-- arXiv Badge -->
        <a href="https://arxiv.org/abs/2510.01284" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
            <span style="background-color: #555; color: white; padding: 4px 8px;">arXiv paper</span>
            <span style="background-color: #b31b1b; color: white; padding: 4px 8px;">2510.01284</span>
        </a>
        
        <!-- Project Page Badge -->
        <a href="https://aaxwaz.github.io/Ovi/" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
            <span style="background-color: #555; color: white; padding: 4px 8px;">Project page</span>
            <span style="background-color: #4c1; color: white; padding: 4px 8px;">More visualizations</span>
        </a>
        
        <!-- Hugging Face Badge -->
        <a href="https://huggingface.co/chetwinlow1/Ovi" target="_blank" style="text-decoration: none; display: inline-flex; border-radius: 4px; overflow: hidden;">
            <span style="background-color: #555; color: white; padding: 4px 8px;">🤗 Hugging Face</span>
            <span style="background-color: #ff9a00; color: white; padding: 4px 8px;">Model</span>
        </a>
    </div>
    <p style="margin-top: 10px; font-size: 0.9em; color: #888;">
        Thank you for trying out Ovi! For more details, please refer to the project page and paper.
    </p>
</div>
"""

# Build UI
with gr.Blocks(css=css) as demo:
    
    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        with gr.TabItem("Ovi", id=0):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(
                        type="filepath", 
                        label="First Frame Image (upload or leave empty for t2v)", 
                        elem_classes="video-size", 
                        elem_id="app-video-player"
                        )
                    if args.use_image_gen:
                        with gr.Accordion("🖼️ Image Generation Options", visible=True):
                            image_text_prompt = gr.Textbox(label="Image Prompt", placeholder="Describe the image you want to generate...")
                            with gr.Row():
                                image_seed = gr.Number(minimum=0, maximum=4294967295, value=42, label="Image Seed")
                                randomize_image_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)
                            image_height = gr.Number(minimum=128, maximum=1280, value=720, step=32, label="Image Height")
                            image_width = gr.Number(minimum=128, maximum=1280, value=1280, step=32, label="Image Width")
                            gen_img_btn = gr.Button("Generate Image 🎨")
                    else:
                        gen_img_btn = None
                        randomize_image_seed_checkbox = None # Ensure it's not referenced if not created
                        
                    run_btn = gr.Button("Generate Video 🚀", variant="primary", size="sm")

                    with gr.Row():
                        video_text_prompt = gr.Textbox(label="Video Prompt", placeholder="Use <S>...<E> to frame speech.  <AUDCAP>...<ENDAUDCAP> can be used at the end of the prompt to frame additional audio features.", lines=6)
                        
                    with gr.Accordion("🎬 Video Generation Options", open=True):
                        with gr.Row():
                            video_seed = gr.Number(minimum=0, maximum=4294967295, value=42, label="Video Seed", precision=0)
                            randomize_video_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)

                        # This group only appears for Text-to-Video
                        with gr.Group(visible=True) as t2v_resolution_group:
                            gr.Markdown("### RESOLUTION (Text-to-Video Only)")
                            with gr.Row():
                                video_height = gr.Slider(minimum=256, maximum=1024, value=512, step=32, label="Video Height")
                                video_width = gr.Slider(minimum=256, maximum=1024, value=960, step=32, label="Video Width")
                            gr.Markdown(
                            "ℹ️ Ovi will generate a video at an optimal resolution guided by your selection, using a total pixel budget."                        
                            )
                        with gr.Group(visible=False) as i2v_info_group:
                            gr.Markdown(
                                "ℹ️ Video will automatically match the input image's aspect ratio, scaled to the selected resolution budget."
                            )
                        sample_steps = gr.Slider(value=50, label="Sample Steps", precision=0, minimum=5, maximum=100, step=1)
                   
                    with gr.Accordion("🎬 Advanced Options", open=False):                    
                        solver_name = gr.Dropdown(
                            choices=["unipc", "euler", "dpm++"], value="unipc", label="Solver Name"
                        )
                        with gr.Row():
                            resolution_budget = gr.Radio(
                                ["Standard (720²)", "High (832²)", "Max (960²)"],
                                value="Standard (720²)",
                                label="Pixel Budget",
                                info="Higher budgets offer more detail but take longer and use more resources."
                            )                      
                        with gr.Row():                        
                            video_duration = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Video Duration (seconds)",
                                info="Ovi was trained for 5s. 7s is viable. Note changing this is experimental and may reduce coherence."
                            )
                        with gr.Row():
                            use_tiled_vae_checkbox = gr.Checkbox(
                                label="Use Tiled VAE Decode", 
                                value=True, 
                                info="Recommended for GPUs with < 24GB VRAM. Disable for faster decoding on high-VRAM cards."
                            )
                            vae_tile_size_slider = gr.Slider( 
                                minimum=16,
                                maximum=128,
                                value=32,
                                step=16,
                                label="VAE Tile Size",
                                info="Increase to improve final decoding speed at the cost of VRAM."
                            )                    
                        shift = gr.Slider(minimum=0.5, maximum=20.0, value=5.0, step=1.0, label="Shift")
                        video_guidance_scale = gr.Slider(minimum=0.5, maximum=10.0, value=4.0, step=0.5, label="Video Guidance Scale")
                        audio_guidance_scale = gr.Slider(minimum=0.5, maximum=10.0, value=3.0, step=0.5, label="Audio Guidance Scale")
                        slg_layer = gr.Number(minimum=-1, maximum=30, value=11, step=1, label="SLG Layer")
                        video_negative_prompt = gr.Textbox(label="Video Negative Prompt", value="jitter, bad hands, blur, distortion")
                        audio_negative_prompt = gr.Textbox(label="Audio Negative Prompt", value="robotic, muffled, echo, distorted")

                with gr.Column():
                    output_path = gr.Video(
                        label="Generated Video",
                        autoplay=True,
                        interactive=False,
                        elem_classes="video-size",
                        elem_id="app-video-player"
                    )     
                    
                    with gr.Group(): 
                        with gr.Row():
                            save_button = gr.Button("Save Manually 💾", size="sm", variant="primary")
                            send_to_toolbox_btn = gr.Button("Send to Toolbox 🛠️", size="sm")
                        autosave_checkbox = gr.Checkbox(label="Autosave Video", value=True)
                        open_folder_button = gr.Button("Open Output Folder 📂", size="sm")
                    with gr.Row():                
                        unload_button = gr.Button("Unload Models 🧹", size="sm", variant="stop")
                    with gr.Row():                    
                        save_status = gr.HTML(value="")
                    with gr.Row():       
                        # Status Info (for cpu/gpu monitor)
                        resource_monitor = gr.Textbox(
                            lines=8,
                            container=False,
                            interactive=False,
                        )                          
                    with gr.Row():
                        generation_console = gr.Textbox(
                            label="Generation Console",
                            interactive=False,
                            lines=12,
                            value="ℹ️ Models will be loaded into RAM on first generation.\nThis will add extra time to your first run, but subsequent generations will be faster.\n\nReady to generate! Click 'Generate Video 🚀' to begin."
                        )

        # --- MODIFIED TOOLBOX TAB ---
        with gr.TabItem("🛠️ Toolbox", id=1):
            with gr.Row():
                # --- Left Column: Inputs and Pipeline Control ---
                with gr.Column(scale=1):
                    # Hidden state to track the active input tab (0=Single, 1=Batch)
                    tb_active_tab_index = gr.Number(value=0, visible=False)
                    
                    with gr.Tabs() as tb_input_tabs:
                        with gr.TabItem("Single Video", id=0):
                             tb_input_video = gr.Video(label="Toolbox Input Video", autoplay=True, elem_classes="video-size", elem_id="toolbox-video-player")
                        with gr.TabItem("Batch Video", id=1):
                            tb_batch_input_files = gr.File(
                                label="Upload Multiple Videos for Batch Processing",
                                file_count="multiple",
                                type="filepath"
                            )
                    
                        with gr.Group():
                            tb_pipeline_steps_chkbox = gr.CheckboxGroup(
                                choices=["Upscale", "Frame Adjust", "Export"],
                                value=[],
                                show_label=False,
                                info="Preconfigure the Operations Settings in the section below and use these checkboxes to run them in order. Note that batch processing requires at least one checkbox checked."
                            )
                            tb_start_pipeline_btn = gr.Button("🚀 Start Pipeline Processing", variant="primary", size="sm")

                # --- Right Column: Output and Controls ---
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("Processed Video"):
                            processed_video = gr.Video(label="Toolbox Processed Video", interactive=False, elem_classes="video-size", elem_id="app-video-player")
                    with gr.Row():
                        tb_use_as_input_btn = gr.Button("Use as Input", size="sm", scale=4)
                        initial_autosave_state = ovi_toolbox_processor.autosave_enabled
                        tb_manual_save_btn = gr.Button("Manual Save 💾", variant="secondary", size="sm", scale=4, visible=not initial_autosave_state)

                    # --- Settings & File Management Group ---
                    with gr.Group():
                        tb_open_folder_btn = gr.Button("📁 Open Outputs", scale=1, variant="huggingface", size="sm")
                        with gr.Row():
                            tb_autosave_checkbox = gr.Checkbox(label="Autosave", scale=1, value=initial_autosave_state)
                            tb_clear_temp_startup_checkbox = gr.Checkbox(label="Clear temp on start", scale=1, value=ovi_toolbox_processor.clear_temp_on_startup)
                        with gr.Row():                               
                            tb_clear_temp_btn = gr.Button("🗑️ Clear Temp", size="sm", scale=1, variant="stop")
                    
                    
            # --- Accordion for all operation settings ---
            with gr.Accordion("Operations Settings", open=True):
                tb_save_settings_btn = gr.Button("💾 Permanently Save ALL Operation Settings to Start-up Defaults", size="sm", variant="primary")
                with gr.Tabs():
                    with gr.TabItem("📈 Upscale (ESRGAN)"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                upscale_model_select = gr.Dropdown(
                                    choices=esrgan_model_choices,
                                    value=default_upscale_model,
                                    label="ESRGAN Model",
                                    info="Select the Real-ESRGAN model."
                                )
                                model_info_display = gr.Textbox(
                                    label="Selected Model Info",
                                    interactive=False,
                                    lines=2,
                                    value=initial_model_info_update.get('value') # Set initial value
                                )
                                upscale_factor_slider = gr.Slider(
                                    minimum=initial_slider_update.get('minimum', 1.0),
                                    maximum=initial_slider_update.get('maximum', 2.0),
                                    value=default_upscale_factor,
                                    step=0.1,
                                    label=initial_slider_update.get('label', "Upscale Factor"),
                                    info="Desired output scale. UI adjusts max value to the model's native scale."
                                )
                            with gr.Column(scale=2):
                                upscale_tile_size_radio = gr.Radio(
                                    choices=[("None (Recommended)", 0), ("512px", 512), ("256px", 256)],
                                    value=default_upscale_tile_size, 
                                    label="Tile Size for Upscaling",
                                    info="Use '512px' or '256px' for out-of-memory errors."
                                )
                                upscale_enhance_face_checkbox = gr.Checkbox(
                                    visible=False,
                                    label="Enhance Faces (GFPGAN)", 
                                    value=default_upscale_enhance_face,
                                    info="Can improve faces. Slow and will overwrite the mouth... don't use for Ovi."
                                )
                                upscale_use_streaming_checkbox = gr.Checkbox(
                                    label="Use Streaming (Low Memory Mode)", 
                                    value=default_upscale_streaming,
                                    info="Enable for slow and stable processing of long or high-res videos."
                                )
                                denoise_strength_slider = gr.Slider(
                                    label="Denoise Strength (for x4v3 model)",
                                    minimum=0.0, maximum=1.0, step=0.01,
                                    value=default_denoise_strength,
                                    info="Adjusts denoising for RealESR-general-x4v3 model only.",
                                    visible=initial_denoise_update.get('visible', False)
                                )
                        upscale_video_btn = gr.Button("🚀 Upscale Video", variant="primary")

                    # --- Frame Adjust Tab ---
                    with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                        with gr.Row():
                            gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                        with gr.Row():
                            process_fps_mode = gr.Radio(
                                choices=["No Interpolation", "2x Frames", "4x Frames"], value=default_fps_mode,  label="RIFE Frame Interpolation",
                                info="Select '2x' or '4x' RIFE Interpolation to double or quadruple the frame rate, creating smoother motion. 4x is more intensive and runs the 2x process twice."
                            )
                            frames_use_streaming_checkbox = gr.Checkbox(
                                label="Use Streaming (Low Memory Mode)", value=default_frames_streaming,
                                info="Enable for stable, low-memory RIFE on long videos. This avoids loading all frames into RAM. Note: 'Adjust Video Speed' is ignored in this mode."              
                            )
                        with gr.Row():
                            process_speed_factor = gr.Slider(
                                minimum=0.5, maximum=2.0, step=0.05, value=default_speed_factor, label="Adjust Video Speed Factor",
                                info="Values < 1.0 slow down the video, values > 1.0 speed it up. Affects video and audio."
                            )
                        process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

                    # --- Export Tab ---
                    with gr.TabItem("📦 Compress, Encode & Export"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                export_format_radio = gr.Radio(
                                    ["MP4", "WebM", "GIF"], value=default_export_format, label="Output Format",
                                    info="MP4 is best for general use. WebM is great for web/Discord (smaller size). GIF is a widely-supported format for short, silent, looping clips. GIF output will always be saved."
                                )
                                export_quality_slider = gr.Slider(
                                    0, 100, value=default_export_quality, step=1, label="Quality",
                                    info="Higher quality means a larger file size. 80-90 is a good balance for MP4/WebM."
                                )
                            with gr.Column(scale=2):
                                export_resize_slider = gr.Slider(
                                    256, 2048, value=default_export_max_width, step=64, label="Max Width (pixels)",
                                    info="Resizes the video to this maximum width while maintaining aspect ratio. A powerful way to reduce file size."
                                )
                                export_name_input = gr.Textbox(
                                    label="Output Filename (optional)",
                                    value=default_export_name,
                                    placeholder="e.g., my_final_video_for_discord",
                                                                    )
                        export_video_btn = gr.Button("🚀 Export Video", variant="primary")
            
            # --- Bottom Row for Status and File Management ---
            with gr.Row():
                tb_status_message = gr.Textbox(label="Toolbox Console", lines=8, interactive=False)
            
    gr.HTML(footer_html)

    ### --- EVENT HANDLERS --- ###
    
    # --- Ovi Tab Handlers ---
    
    def set_t2v_mode():
        """Called when image is cleared. Shows T2V controls, hides I2V info."""
        return {
            t2v_resolution_group: gr.update(visible=True),
            i2v_info_group: gr.update(visible=False)
        }

    def set_i2v_mode():
        """Called when image is uploaded. Hides T2V controls, shows I2V info."""
        return {
            t2v_resolution_group: gr.update(visible=False),
            i2v_info_group: gr.update(visible=True)
        }
        
    # Attach the functions to the image component's events
    image.clear(fn=set_t2v_mode, inputs=None, outputs=[t2v_resolution_group, i2v_info_group])
    image.upload(fn=set_i2v_mode, inputs=None, outputs=[t2v_resolution_group, i2v_info_group])

    run_btn.click(
        fn=prepare_video_seed,
        inputs=[randomize_video_seed_checkbox, video_seed],
        outputs=[video_seed]
    ).then(
        fn=generate_video,
        inputs=[
            video_text_prompt, image, video_height, video_width,
            video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, video_negative_prompt, audio_negative_prompt, autosave_checkbox,
            resolution_budget, video_duration,
            use_tiled_vae_checkbox, vae_tile_size_slider
        ],
        outputs=[output_path, generation_console]
    )
    
    def send_to_toolbox(video_path):
        if not video_path:
            return gr.update(), gr.update(), '<div style="padding: 8px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404;">⚠️ No video to send!</div>'
        # Switches to tab 1 (Toolbox) and sets the input video value
        return gr.update(selected=1), gr.update(value=video_path), '<div style="padding: 8px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724;">✅ Video sent to Toolbox!</div>'

    send_to_toolbox_btn.click(
        fn=send_to_toolbox,
        inputs=[output_path],
        outputs=[main_tabs, tb_input_video, save_status]
    )
    save_button.click(
        fn=save_video_manually,
        inputs=[output_path],
        outputs=[save_status]
    )    
    # --- Toolbox Tab Handlers ---

    def handle_single_operation(operation_func, video_path, status_message, **kwargs):
        if not video_path:
            return None, "⚠️ No input video found."

        temp_video = operation_func(video_path, progress=gr.Progress(), **kwargs)

        if not temp_video or temp_video == video_path:
            return video_path, f"❌ {status_message} failed. Check console."

        final_video_path = temp_video
        message = f"✅ {status_message} complete."

        if ovi_toolbox_processor.autosave_enabled:
            # Don't add "_final" suffix - preserve the filename from the operation
            temp_path = Path(temp_video)
            final_path = ovi_toolbox_processor.output_dir / temp_path.name
            final_video_path = ovi_toolbox_processor._copy_to_permanent_storage(temp_video, final_path)
            message += f"\n✅ Autosaved result to: {final_path}"
        else:
            message += "\nℹ️ Autosave is off. Result is temporary. Use 'Manual Save'."
        
        return final_video_path, message

    def update_active_tab_index(evt: gr.SelectData):
        return evt.index
    tb_input_tabs.select(fn=update_active_tab_index, inputs=None, outputs=[tb_active_tab_index])
    
    tb_start_pipeline_btn.click(
        fn=handle_start_pipeline,
        inputs=[
            tb_active_tab_index, tb_input_video, tb_batch_input_files, tb_pipeline_steps_chkbox,
            upscale_model_select, upscale_factor_slider, upscale_tile_size_radio,
            upscale_enhance_face_checkbox, denoise_strength_slider, upscale_use_streaming_checkbox,
            process_fps_mode, process_speed_factor, frames_use_streaming_checkbox,
            export_format_radio, export_quality_slider, export_resize_slider, export_name_input
        ],
        outputs=[processed_video, tb_status_message]
    )

    def use_as_input(processed_video_path):
        if not processed_video_path:
            return gr.update(), "⚠️ No processed video available."
        return processed_video_path, "✅ Moved processed video to input."
    tb_use_as_input_btn.click(
        fn=use_as_input,
        inputs=[processed_video],
        outputs=[tb_input_video, tb_status_message]
    )

    def handle_autosave_toggle(is_enabled):
        message = ovi_toolbox_processor.set_autosave_mode(is_enabled)
        return gr.update(visible=not is_enabled), message
    tb_autosave_checkbox.change(
        fn=handle_autosave_toggle,
        inputs=[tb_autosave_checkbox],
        outputs=[tb_manual_save_btn, tb_status_message]
    )
    
    upscale_model_select.change(
        fn=ovi_toolbox_processor.get_model_info_and_update_scale_slider,
        inputs=[upscale_model_select],
        outputs=[model_info_display, upscale_factor_slider, denoise_strength_slider]
    )
    
    # --- Corrected Single Operation Buttons (No State) ---
    upscale_video_btn.click(
        lambda video_path, status, model, scale, tile, face, denoise, stream: handle_single_operation(ovi_toolbox_processor.upscale_video, video_path, status, model_key=model, scale_factor=scale, tile_size=tile, enhance_face=face, denoise_strength=denoise, use_streaming=stream),
        inputs=[tb_input_video, gr.Textbox("Upscaling", visible=False), upscale_model_select, upscale_factor_slider, upscale_tile_size_radio, upscale_enhance_face_checkbox, denoise_strength_slider, upscale_use_streaming_checkbox],
        outputs=[processed_video, tb_status_message]
    )
    process_frames_btn.click(
        lambda video_path, status, fps, speed, stream: handle_single_operation(ovi_toolbox_processor.adjust_frames, video_path, status, fps_mode=fps, speed_factor=speed, use_streaming=stream),
        inputs=[tb_input_video, gr.Textbox("Frame Adjustment", visible=False), process_fps_mode, process_speed_factor, frames_use_streaming_checkbox],
        outputs=[processed_video, tb_status_message]
    )
    export_video_btn.click(
        lambda video_path, status, format, quality, width, name: handle_single_operation(ovi_toolbox_processor.export_video, video_path, status, export_format=format, quality=quality, max_width=width, output_name=name),
        inputs=[tb_input_video, gr.Textbox("Exporting", visible=False), export_format_radio, export_quality_slider, export_resize_slider, export_name_input],
        outputs=[processed_video, tb_status_message]
    )

    # Manual Save Button - Reverted to your proven, simple logic
    def handle_manual_save(video_path_from_player):
        if not video_path_from_player or not os.path.exists(video_path_from_player):
             return "⚠️ No video in the output player to save."
        
        saved_path = ovi_toolbox_processor.save_video_from_any_source(video_path_from_player)
        
        if saved_path:
            return f"✅ Video successfully saved to: {saved_path}"
        else:
            return "❌ An error occurred during save. Check the console for details."

    tb_manual_save_btn.click(
        fn=handle_manual_save,
        inputs=[processed_video], # Takes input directly from the video player
        outputs=[tb_status_message]  # Only needs to update the status message
    )

    # File management buttons
    tb_open_folder_btn.click(fn=ovi_toolbox_processor.open_output_folder, outputs=[tb_status_message])
    tb_clear_temp_btn.click(fn=ovi_toolbox_processor.clear_temp_folder, outputs=[tb_status_message])

    # Settings management
    def toggle_clear_temp_startup(enabled):
        ovi_toolbox_processor.clear_temp_on_startup = enabled
        ovi_toolbox_processor.save_setting("clear_temp_on_startup", enabled)
        return f"✅ Clear temp on startup: {'ENABLED' if enabled else 'DISABLED'}"
    
    def save_current_settings(upscale_model, upscale_factor, upscale_tile_size, upscale_enhance_face, 
                             upscale_streaming, denoise_strength, fps_mode, speed_factor, 
                             frames_streaming, export_format, export_quality, export_max_width, export_name):
        current_settings = {
            "autosave_enabled": ovi_toolbox_processor.autosave_enabled,
            "clear_temp_on_startup": ovi_toolbox_processor.clear_temp_on_startup,
            "upscale_model": upscale_model,
            "upscale_factor": upscale_factor,
            "upscale_tile_size": upscale_tile_size,
            "upscale_enhance_face": upscale_enhance_face,
            "upscale_use_streaming": upscale_streaming,
            "denoise_strength": denoise_strength,
            "fps_mode": fps_mode,
            "speed_factor": speed_factor,
            "frames_use_streaming": frames_streaming,
            "export_format": export_format,
            "export_quality": export_quality,
            "export_max_width": export_max_width,
            "export_name": export_name,
        }
        return ovi_toolbox_processor.save_all_settings(current_settings)
    
    tb_clear_temp_startup_checkbox.change(
        fn=toggle_clear_temp_startup,
        inputs=[tb_clear_temp_startup_checkbox],
        outputs=[tb_status_message]
    )
    
    tb_save_settings_btn.click(
        fn=save_current_settings,
        inputs=[upscale_model_select, upscale_factor_slider, upscale_tile_size_radio, upscale_enhance_face_checkbox, upscale_use_streaming_checkbox, denoise_strength_slider, process_fps_mode, process_speed_factor, frames_use_streaming_checkbox, export_format_radio, export_quality_slider, export_resize_slider, export_name_input],
        outputs=[tb_status_message]
    )
    
    def update_monitor():
        return SystemMonitor.get_system_info()
        
    monitor_timer = gr.Timer(2, active=True)
    monitor_timer.tick(fn=update_monitor, outputs=resource_monitor) 
    
    open_folder_button.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[save_status]
    )
    unload_button.click(
        fn=unload_models_from_ram,
        inputs=[],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(
    server_name=args.server_name, 
    server_port=args.server_port,
    share=args.share, 
    mcp_server=args.mcp_server,
)