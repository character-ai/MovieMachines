import gradio as gr
import torch
import argparse
from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from diffusers import FluxPipeline
import tempfile
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text, scale_hw_to_area_divisible
import os
import uuid
import random
from datetime import datetime
import subprocess
import platform
import shutil
from PIL import Image
from huggingface_hub import snapshot_download

# Try to suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    import socket # Required for the ConnectionResetError
    
    # This is the most important part of the fix. It replaces the default
    # Windows event loop with a more compatible one.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # This part "monkey-patches" a low-level asyncio function to silence
    # errors that are benign but noisy.
    def silence_connection_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError):
                # Silences the "[WinError 10054]..." and "[WinError 32]..." errors.
                # These are common when the client (browser) disconnects abruptly.
                pass
            except RuntimeError as e:
                # Silences the "Event loop is closed" error.
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    # Apply the patch to the function that causes the error
    # We need to import the class directly to patch its method
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
    "--cpu_offload",
    action="store_true",
    help="Enable CPU offload for both OviFusionEngine and FluxPipeline"
)
parser.add_argument(
    "--fp8",
    action="store_true",
    help="Enable 8 bit quantization of the fusion model",
)
args = parser.parse_args()

# ----------------------------
# Model Download Logic
# ----------------------------
def check_and_download_models():
    """
    Checks for model files and downloads them from Hugging Face if missing.
    """
    base_ckpt_dir = "./ckpts"
    
    # This dictionary structure remains the same
    models_to_check = [
        {
            "name": "Wan2.2-TI2V-5B",
            "repo_id": "Wan-AI/Wan2.2-TI2V-5B",
            "local_dir": os.path.join(base_ckpt_dir, "Wan2.2-TI2V-5B"),
            "allow_patterns": [
                "google/*", # <-- Add this line back
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
        {
            "name": "Ovi (FP8 Quantized)",
            "repo_id": "rkfg/Ovi-fp8_quantized",
            "local_dir": os.path.join(base_ckpt_dir, "Ovi"),
            "allow_patterns": ["*.safetensors"], 
            "check_files": [os.path.join(base_ckpt_dir, "Ovi", "model_fp8_e4m3fn.safetensors")]
        }
    ]
    
    print()
    print("--- Checking for required model weights ---")
    for model in models_to_check:
        # print(f"Checking model: '{model['name']}'")
        all_files_exist = True
        
        for file_path in model["check_files"]:
            if os.path.exists(file_path):
                pass
            else:
                print(f"    ❌ Missing file: {file_path}")
                all_files_exist = False

        # If all files were found, continue to the next model
        if all_files_exist:
            print(f"✅ '{model['name']}' is present.")
            continue

        # If any file was missing, start the download
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
enable_cpu_offload = args.cpu_offload or args.use_image_gen
use_image_gen = args.use_image_gen
fp8 = args.fp8
print(f"loading model... {enable_cpu_offload=}, {use_image_gen=}, {fp8=} for gradio demo")
DEFAULT_CONFIG["cpu_offload"] = (
    enable_cpu_offload  # always use cpu offload if image generation is enabled
)
DEFAULT_CONFIG["mode"] = "t2v"  # hardcoded since it is always cpu offloaded
DEFAULT_CONFIG["fp8"] = fp8
ovi_engine = OviFusionEngine()
flux_model = None

# if fp8:
    # assert not use_image_gen, "Image generation with FluxPipeline is not supported with fp8 quantization. This is because if you are unable to run the bf16 model, you likely cannot run image gen model"
    
if use_image_gen:
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
    flux_model.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM
print("loaded model")


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
):
    try:
        image_path = None
        if image is not None:
            image_path = image

        video_frame_height_width = [video_height, video_width]
        
        generated_video, generated_audio, _ = ovi_engine.generate(
            text_prompt=text_prompt,
            image_path=image_path,
            video_frame_height_width=video_frame_height_width,
            seed=video_seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            slg_layer=slg_layer,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
        )

        # file naming with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Ovi_{timestamp}.mp4"
        
        # Determine save path based on autosave
        if autosave_video:
            output_path = os.path.join("outputs", output_filename)
        else:
            # Use a temporary directory for non-autosaved files
            output_path = os.path.join(tempfile.gettempdir(), output_filename)

        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)

        # Return the path and the updated seed value
        return output_path

    except Exception as e:
        print(f"Error during video generation: {e}")
        return None, video_seed

def prepare_video_seed(randomize_seed, current_seed):
    """
    Determines the seed for the upcoming generation.
    This runs instantly before the main video generation starts.
    """
    if randomize_seed:
        return random.randint(0, 100000)
    return current_seed
    
def generate_image(text_prompt, image_seed, randomize_image_seed, image_height, image_width):
    if flux_model is None:
        return None, image_seed
        
    # If randomize seed is checked, generate a random seed
    if randomize_image_seed:
        image_seed = random.randint(0, 100000)

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
        return f"Video saved to {destination}"
    return "No video to save or file not found."

def open_output_folder():
    folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin": # macOS
        subprocess.Popen(["open", folder_path])
    else: # Linux
        subprocess.Popen(["xdg-open", folder_path])
    return "Opened output folder."


css = """
.video-size video, 
.video-size img {
    max-height: 60vh;
    object-fit: contain;
}

/* hide the gr.Video source selection bar for tb_input_video_component */
#app-video-player .source-selection {
    display: none !important;
}
#start-btn {
    margin-top: -14px !important; /* Adjust this value to get the perfect alignment */
}        
        
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
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs(elem_id="output_tabs"):
                with gr.TabItem("Input Image"):
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
                            image_seed = gr.Number(minimum=0, maximum=100000, value=42, label="Image Seed")
                            randomize_image_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)
                        image_height = gr.Number(minimum=128, maximum=1280, value=720, step=32, label="Image Height")
                        image_width = gr.Number(minimum=128, maximum=1280, value=1280, step=32, label="Image Width")
                        gen_img_btn = gr.Button("Generate Image 🎨")
                else:
                    gen_img_btn = None
                    randomize_image_seed_checkbox = None # Ensure it's not referenced if not created
                    
                run_btn = gr.Button("Generate Video 🚀", variant="primary", size="sm", elem_id="start-btn")

                with gr.Row():
                    video_text_prompt = gr.Textbox(label="Video Prompt", placeholder="Use <S>...<E> to frame speech.  <AUDCAP>...<ENDAUDCAP> can be used at the end of the prompt to frame additional audio features.", lines=6)
                    
                with gr.Accordion("🎬 Video Generation Options", open=True):
                    with gr.Row():
                        video_seed = gr.Number(minimum=0, maximum=100000, value=42, label="Video Seed")
                        randomize_video_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)

                    # This group only appears for Text-to-Video
                    with gr.Group(visible=True) as t2v_resolution_group:
                        gr.Markdown("### RESOLUTION (Text-to-Video Only)")
                        with gr.Row():
                            video_height = gr.Slider(minimum=256, maximum=1024, value=512, step=32, label="Video Height")
                            video_width = gr.Slider(minimum=256, maximum=1024, value=960, step=32, label="Video Width")
                        gr.Markdown(
                        "ℹ️ Ovi will generate a video at an optimal resolution guided by your selection, using a total budget of 720x720 pixels."                        
                        )
                    with gr.Group(visible=False) as i2v_info_group:
                        gr.Markdown(
                            "ℹ️ Video will automatically match the input image's aspect ratio."
                        )
                        
                    sample_steps = gr.Slider(value=50, label="Sample Steps", precision=0, minimum=5, maximum=100, step=1)
               
                with gr.Accordion("🎬 Advanced Options", open=False):                    
                    solver_name = gr.Dropdown(
                        choices=["unipc", "euler", "dpm++"], value="unipc", label="Solver Name"
                    )

                    shift = gr.Slider(minimum=0.5, maximum=20.0, value=5.0, step=1.0, label="Shift")
                    video_guidance_scale = gr.Slider(minimum=0.5, maximum=10.0, value=4.0, step=0.5, label="Video Guidance Scale")
                    audio_guidance_scale = gr.Slider(minimum=0.5, maximum=10.0, value=3.0, step=0.5, label="Audio Guidance Scale")
                    slg_layer = gr.Number(minimum=-1, maximum=30, value=11, step=1, label="SLG Layer")
                    video_negative_prompt = gr.Textbox(label="Video Negative Prompt", value="jitter, bad hands, blur, distortion")
                    audio_negative_prompt = gr.Textbox(label="Audio Negative Prompt", value="robotic, muffled, echo, distorted")


        with gr.Column():
            with gr.Tabs(elem_id="output_tabs"):
                with gr.TabItem("Video Output"):
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
                    open_folder_button = gr.Button("Open Output Folder 📂", size="sm")
                autosave_checkbox = gr.Checkbox(label="Autosave Video", value=True)           
            save_status = gr.Textbox(label="Status", interactive=False, lines=2)

    with gr.Row():
        with gr.Accordion("🎬 Instructions & Prompting Guide", open=False):
            gr.Markdown(
                """
                ## 📘 How to Generate a Video

                * **Choose Your Mode** — Upload an image for **Image-to-Video**, or leave it blank for pure **Text-to-Video**.  
                *(The resolution sliders will appear automatically for Text-to-Video mode.)*  
                * **Write Your Prompt** — Describe the scene and action. Use the special tags below to add speech and describe audio. **Getting the prompt format right is the key to a good result!**
                * **Adjust Options** — Fine-tune your video using the "Video Generation Options" and "Advanced Options".
                * **Generate** — Click the **Generate Video 🚀** button and wait for your creation to appear.
                
                ---
                ### 💡 Prompting Guide: Adding Speech & Sound

                The model requires special tags in your prompt to generate speech and audio correctly. Follow this structure:
                
                - **For Spoken Dialogue:** Wrap any text you want a character to say in `<S>` and `<E>` tags.
                  - **Format:** `<S>This is the speech content.<E>`
                
                - **For Audio Descriptions:** At the very **end** of your prompt, describe the voice, sound effects, music or ambiance. Wrap this description in `<AUDCAP>` and `<ENDAUDCAP>` tags.
                  - **Format:** `...end of video description. <AUDCAP>Description of all audio, voices, and sound effects.<ENDAUDCAP>`
                  
                - Simply prompting the wrapped speech text, without any additional descriptive prompting, is the most reliable way to animate your character.

                ---
                ###  examples:

                **Example 1: Dialogue between characters**
                - Three men stand facing each other in a room... The man on the left... gestures with his hands as he speaks, `<S>`This world is ours to keep.`<E>` He continues, looking towards the man on the right, `<S>`Humanity endures beyond your code.`<E>` ... Light blue armchairs are visible in the soft-lit background on both sides.. <AUDCAP>Clear male voices speaking, room ambience.<ENDAUDCAP>
                
                **Example 2: Single line of speech with sound effects**
                - Two women stand facing each other in what appears to be a backstage dressing room... The woman on the right... looks back with a pleading or concerned expression, her lips slightly parted as she speaks: `<S>`Humans fight for freedom tonight.`<E>` As she finishes, the woman on the left turns her head away, breaking eye contact.. <AUDCAP>Soft vocal exhalation, female speech, loud abrupt buzzing sound.<ENDAUDCAP>

                **Example 3: Somber tone with ambient sound**
                - The scene is set in a dimly lit, hazy room, creating a somber atmosphere... The woman looks directly forward as she slowly enunciates, `<S>`Only through death will the third door be found`<E>`. The scene ends abruptly.. <AUDCAP>Clear, deliberate female voice speaking, low ambient hum and subtle atmospheric sounds creating a tense mood.<ENDAUDCAP>

                ---
                ### ✨ General Tips
                - Do not be discouraged by weird results. Check your prompt format and try different **Video Seeds** for variety.
                - Experiment with the **Video/Audio Guidance Scale** and **SLG Layer** in the "Advanced Options" to influence how strongly the model follows your prompt.
                """
            )

    gr.HTML(footer_html)
    
    # if args.use_image_gen and gen_img_btn is not None and randomize_image_seed_checkbox is not None:
        # gen_img_btn.click(
            # fn=generate_image,
            # inputs=[image_text_prompt, image_seed, randomize_image_seed_checkbox, image_height, image_width],
            # outputs=[image, image_seed], # Also update the seed number in the UI
        # )

    # Event handlers for showing/hiding the T2V resolution sliders
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
            video_text_prompt, image, video_height, video_width, # <-- ADD video_height and video_width here
            video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, video_negative_prompt, audio_negative_prompt, autosave_checkbox
        ],
        outputs=[output_path]
    )

    save_button.click(
        fn=save_video_manually,
        inputs=[output_path],
        outputs=[save_status]
    )
    
    open_folder_button.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[save_status]
    )
    
if __name__ == "__main__":
    demo.launch(share=False)