import gradio as gr
import torch
import argparse
from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from diffusers import FluxPipeline
import tempfile
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text, scale_hw_to_area_divisible

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


# Initialize OviFusionEngine
enable_cpu_offload = args.cpu_offload or args.use_image_gen
use_image_gen = args.use_image_gen
fp8 = args.fp8
print(f"loading model... {enable_cpu_offload=}, {use_image_gen=}, {fp8=} for gradio demo")
DEFAULT_CONFIG["cpu_offload"] = (
    enable_cpu_offload  # always use cpu offload if image generation is enabled
)
DEFAULT_CONFIG["mode"] = "t2v"  # hardcoded since it is always cpu offloaded
ovi_engine = OviFusionEngine(fp8=fp8)
flux_model = None
if use_image_gen:
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
    flux_model.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM
print("loaded model")


def generate_video(
    text_prompt,
    image,
    video_frame_height,
    video_frame_width,
    video_seed,
    solver_name,
    sample_steps,
    shift,
    video_guidance_scale,
    audio_guidance_scale,
    slg_layer,
    video_negative_prompt,
    audio_negative_prompt,
):
    try:
        image_path = None
        if image is not None:
            image_path = image

        generated_video, generated_audio, _ = ovi_engine.generate(
            text_prompt=text_prompt,
            image_path=image_path,
            video_frame_height_width=[video_frame_height, video_frame_width],
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

        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = tmpfile.name
        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)

        return output_path
    except Exception as e:
        print(f"Error during video generation: {e}")
        return None


def generate_image(text_prompt, image_seed, image_height, image_width):
    if flux_model is None:
        return None
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

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmpfile.name)
    return tmpfile.name


# Build UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Ovi Joint Video + Audio Generation Demo")
    gr.Markdown(
        """
        ## 📘 Instructions

        Follow the steps in order:

        1️⃣ **Enter a Text Prompt** — describe your video. (This text prompt will be shared for image generation if enabled.)  
        2️⃣ **Upload or Generate an Image** — Upload an image or generate one if image generation is enabled.  (If you do not see the image generation options, make sure to run the script with `--use_image_gen`.)  
        3️⃣ **Configure Video Options** — set resolution, seed, solver, and other parameters. (It will automatically use the uploaded/generated image as the first frame, whichever is rendered on your screen at the time of video generation.)  
        4️⃣ **Generate Video** — click the button to produce your final video with audio.  
        5️⃣ **View the Result** — your generated video will appear below.  

        ---

        ### 💡 Tips
        1. For best results, use detailed and specific text prompts.  
        2. Ensure text prompt format is correct, i.e speech to be said should be wrapped with `<S>...<E>`. Can provide optional audio description at the end, wrapping them in `<AUDCAP> ... <ENDAUDCAP>`, refer to examples  
        3. Do not be discouraged by bad or weird results, check prompt format and try different seeds, cfg values and slg layers.
        """
    )


    with gr.Row():
        with gr.Column():
            # Image section
            image = gr.Image(type="filepath", label="First Frame Image (upload or generate)")

            if args.use_image_gen:
                with gr.Accordion("🖼️ Image Generation Options", visible=True):
                    image_text_prompt = gr.Textbox(label="Image Prompt", placeholder="Describe the image you want to generate...")
                    image_seed = gr.Number(minimum=0, maximum=100000, value=42, label="Image Seed")
                    image_height = gr.Number(minimum=128, maximum=1280, value=720, step=32, label="Image Height")
                    image_width = gr.Number(minimum=128, maximum=1280, value=1280, step=32, label="Image Width")
                    gen_img_btn = gr.Button("Generate Image 🎨")
            else:
                gen_img_btn = None

            with gr.Accordion("🎬 Video Generation Options", open=True):
                video_text_prompt = gr.Textbox(label="Video Prompt", placeholder="Describe your video...")
                video_height = gr.Number(minimum=128, maximum=1280, value=512, step=32, label="Video Height")
                video_width = gr.Number(minimum=128, maximum=1280, value=992, step=32, label="Video Width")

                video_seed = gr.Number(minimum=0, maximum=100000, value=100, label="Video Seed")
                solver_name = gr.Dropdown(
                    choices=["unipc", "euler", "dpm++"], value="unipc", label="Solver Name"
                )
                sample_steps = gr.Number(
                    value=50,
                    label="Sample Steps",
                    precision=0,
                    minimum=20,
                    maximum=100
                )
                shift = gr.Slider(minimum=0.0, maximum=20.0, value=5.0, step=1.0, label="Shift")
                video_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=4.0, step=0.5, label="Video Guidance Scale")
                audio_guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.5, label="Audio Guidance Scale")
                slg_layer = gr.Number(minimum=-1, maximum=30, value=11, step=1, label="SLG Layer")
                video_negative_prompt = gr.Textbox(label="Video Negative Prompt", placeholder="Things to avoid in video")
                audio_negative_prompt = gr.Textbox(label="Audio Negative Prompt", placeholder="Things to avoid in audio")

                run_btn = gr.Button("Generate Video 🚀")

        with gr.Column():
            output_path = gr.Video(label="Generated Video")

    if args.use_image_gen and gen_img_btn is not None:
        gen_img_btn.click(
            fn=generate_image,
            inputs=[image_text_prompt, image_seed, image_height, image_width],
            outputs=[image],
        )

    # Hook up video generation
    run_btn.click(
        fn=generate_video,
        inputs=[
            video_text_prompt, image, video_height, video_width, video_seed, solver_name,
            sample_steps, shift, video_guidance_scale, audio_guidance_scale,
            slg_layer, video_negative_prompt, audio_negative_prompt,
        ],
        outputs=[output_path],
    )

if __name__ == "__main__":
    demo.launch(share=True)
