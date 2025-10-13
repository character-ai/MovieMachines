import os
import gc
import sys
import subprocess
import types
import torch
import re
import numpy as np
import imageio
import gradio as gr
import shutil
import traceback
import math
import json
from datetime import datetime
from pathlib import Path

# Local devicetorch import
import devicetorch

from torchvision.transforms.functional import rgb_to_grayscale

# --- Patch for basicsr (must run after torchvision import) ---
functional_tensor_mod = types.ModuleType('functional_tensor')
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor_mod)

# Local imports for RIFE and ESRGAN
from toolbox.rife_core import RIFEHandler
from toolbox.esrgan_core import ESRGANUpscaler

device_name_str = devicetorch.get(torch)

class OviToolboxProcessor:
    """
    A processor for handling upscale, frame adjustment, and export operations.
    """
    def __init__(self):
        self.device_obj = torch.device(device_name_str)
        self.output_dir = Path("outputs/toolbox")
        self.temp_dir = Path("tmp/toolbox")
        self.settings_file = Path("toolbox_settings.json")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.ffmpeg_exe, self.ffprobe_exe, self.has_ffmpeg = self._initialize_ffmpeg()

        self.rife_handler = RIFEHandler()
        self.esrgan_upscaler = ESRGANUpscaler(self.device_obj)
        
        self.autosave_enabled = self.load_setting("autosave_enabled", True)
        self.clear_temp_on_startup = self.load_setting("clear_temp_on_startup", False)

    def load_setting(self, key, default):
        """Loads a single setting from the JSON file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f: return json.load(f).get(key, default)
        except (json.JSONDecodeError, IOError): pass
        return default

    def save_setting(self, key, value):
        """Saves a single setting to the JSON file."""
        settings = {}
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f: settings = json.load(f)
        except (json.JSONDecodeError, IOError): pass
        settings[key] = value
        with open(self.settings_file, 'w') as f: json.dump(settings, f, indent=4)

    def set_autosave_mode(self, is_enabled):
        """Updates the autosave mode and saves it."""
        self.autosave_enabled = is_enabled
        self.save_setting("autosave_enabled", is_enabled)
        status = "ON" if is_enabled else "OFF"
        print(f"Autosave is now {status}.")
        return f"✅ Autosave is now {status}."

    def save_video_from_any_source(self, video_source_path):
        """
        Copies a video from the toolbox temp directory to the permanent output folder,
        preserving its filename. This is the backend for the Manual Save button.
        """
        try:
            # Get the filename from the source path
            source_filename = Path(video_source_path).name
            
            # Create the destination path
            destination_path = self.output_dir / source_filename
            
            print(f"Copying video from '{video_source_path}' to '{destination_path}'")
            
            # Copy the file to preserve it in temp for further operations
            shutil.copy2(video_source_path, destination_path)
            
            return str(destination_path)
            
        except Exception as e:
            print(f"Error during manual save: {e}\n{traceback.format_exc()}")
            return None
            
    def clear_temp_folder(self):
        """Deletes all files and subfolders in the temporary directory."""
        deleted_count, error_count = 0, 0
        if not os.path.exists(self.temp_dir): return "✅ Temporary folder does not exist. Nothing to clear."
        for item in os.listdir(self.temp_dir):
            item_path = os.path.join(self.temp_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path): os.unlink(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path)
                deleted_count += 1
            except Exception as e: print(f"Failed to delete {item_path}: {e}"); error_count += 1
        if error_count > 0: return f"⚠️ Cleared {deleted_count} items from temp folder with {error_count} errors."
        return f"✅ Successfully cleared {deleted_count} items from temporary folder."

    def open_output_folder(self):
        """Opens the toolbox output folder in the system's file explorer."""
        folder_path = os.path.abspath(self.output_dir)
        try:
            if sys.platform == "win32": os.startfile(folder_path)
            elif sys.platform == "darwin": subprocess.Popen(["open", folder_path])
            else: subprocess.Popen(["xdg-open", folder_path])
            return f"Opened output folder: {folder_path}"
        except Exception as e: return f"❌ Error opening folder: {e}"

    def save_all_settings(self, settings_dict):
        """Saves all settings to the JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=4)
            return "✅ Settings saved successfully!"
        except Exception as e:
            return f"❌ Error saving settings: {e}"

    def load_all_settings(self):
        """Loads all settings from the JSON file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def get_model_info_and_update_scale_slider(self, model_key: str):
        """
        Helper function to update UI based on selected ESRGAN model.
        """
        if not model_key:
            return gr.update(value="Info: Select a model."), gr.update(), gr.update(visible=False)

        details = self.esrgan_upscaler.supported_models.get(model_key, {})
        native_scale = details.get('scale', 2.0)
        description = details.get('description', 'No description available.')

        info_update = gr.update(value=description)
        slider_update = gr.update(
            minimum=1.0, maximum=native_scale, value=native_scale,
            label=f"Target Upscale Factor (Native {native_scale}x)"
        )
        denoise_visible = "RealESR-general-x4v3" in model_key
        denoise_update = gr.update(visible=denoise_visible, value=0.5)

        return info_update, slider_update, denoise_update
        
    def apply_startup_clear_temp(self):
        """Clears temp folder on startup if enabled."""
        if self.clear_temp_on_startup:
            print("--- Clearing Toolbox Temp Folder on Startup ---")
            clear_message = self.clear_temp_folder()
            # print(clear_message)
            # print("---------------------------------------------")
            return clear_message
        return None
        
    def _initialize_ffmpeg(self):
        """Finds FFmpeg/FFprobe and sets status flags."""
        ffmpeg_path, ffprobe_path = self._find_ffmpeg_executables()
        has_ffmpeg = bool(ffmpeg_path) and bool(ffprobe_path)
        if not has_ffmpeg: print("WARNING: FFmpeg or FFprobe not found. Audio handling and some export formats will be disabled.")
        return ffmpeg_path, ffprobe_path, has_ffmpeg

    def _find_ffmpeg_executables(self):
        """Finds ffmpeg and ffprobe, prioritizing system PATH then imageio."""
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        ffprobe_path = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        if not ffmpeg_path:
            try:
                imageio_ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
                if os.path.isfile(imageio_ffmpeg_exe): ffmpeg_path = imageio_ffmpeg_exe
            except Exception: pass
        return ffmpeg_path, ffprobe_path

    def _clean_filename(self, filename):
        """
        Intelligently handles timestamps in filenames.
        - Preserves original timestamps from unprocessed files
        - For processed files, removes only the timestamp added by the most recent operation
        """
        # Check if this filename already contains operation suffixes
        operation_patterns = ['upscaled_', 'frames_', 'exported_']
        has_operations = any(pattern in filename for pattern in operation_patterns)
        
        if has_operations:
            # Find all timestamps in the filename
            timestamp_pattern = r'_\d{8}_\d{6}'
            timestamps = re.findall(timestamp_pattern, filename)
            
            if len(timestamps) > 1:
                # Multiple timestamps - remove only the last one (most recent operation)
                # Replace the last timestamp with empty string
                last_timestamp = timestamps[-1]
                filename = filename.replace(last_timestamp, '')
            elif len(timestamps) == 1:
                # Only one timestamp - this might be original, but since we have operations,
                # it's likely from a previous operation, so remove it
                filename = re.sub(timestamp_pattern, '', filename)
        # If no operations yet, preserve all timestamps (original file)
        
        return filename.strip('_')

    def _generate_output_path(self, input_path, suffix, ext=".mp4", is_temp=False, batch_folder=None):
        """Generates a unique output path for processed videos, adapted from reference code."""
        base_name = self._clean_filename(Path(input_path).stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{suffix}_{timestamp}{ext}"
        if is_temp: return self.temp_dir / filename
        if batch_folder:
            target_dir = self.output_dir / batch_folder
            os.makedirs(target_dir, exist_ok=True)
            return target_dir / filename
        return self.output_dir / filename

    def _copy_to_permanent_storage(self, temp_path, final_path):
        """Copies a temp file to permanent storage and cleans up the source temp file."""
        try:
            shutil.copy(temp_path, final_path)
            os.remove(temp_path)
            return str(final_path)
        except Exception as e:
            print(f"Error moving file to permanent storage: {e}")
            return str(temp_path)

    def _get_video_frame_count(self, video_path):
        """Uses ffprobe to get an accurate frame count."""
        if not self.has_ffmpeg: return None
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path]
            return int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
        except Exception: return None

    def _has_audio_stream(self, video_path):
        """Checks if a video file has an audio stream using ffprobe."""
        if not self.has_ffmpeg: return False
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path]
            return "audio" in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip().lower()
        except Exception: return False

    def upscale_video(self, video_path, model_key, scale_factor, tile_size, enhance_face, denoise_strength, use_streaming, progress=gr.Progress()):
        if not video_path: print("No input video for upscaling."); return None
        
        temp_video_path, writer, reader = None, None, None # Initialize all to None
        try:
            print(f"Starting upscale: {model_key} at {scale_factor}x, Streaming: {use_streaming}")
            upsampler = self.esrgan_upscaler.load_model(model_key, tile_size, denoise_strength)
            if enhance_face: self.esrgan_upscaler._load_face_enhancer(bg_upsampler=upsampler)
            
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']
            
            temp_video_path = self._generate_output_path(video_path, f"upscaled_{model_key}_temp", is_temp=True)
            
            if use_streaming:
                writer = imageio.get_writer(temp_video_path, fps=fps, quality=8)
                total_frames = self._get_video_frame_count(video_path) or len(reader)
                for i in progress.tqdm(range(total_frames), desc="Upscaling Frames (Streaming)"):
                    frame = reader.get_data(i)
                    upscaled_frame = self.esrgan_upscaler.upscale_frame(frame, model_key, scale_factor, enhance_face)
                    writer.append_data(upscaled_frame)
            else:
                frames = [frame for frame in progress.tqdm(reader, desc="Reading Frames")]
                upscaled_frames = [self.esrgan_upscaler.upscale_frame(frame, model_key, scale_factor, enhance_face) for frame in progress.tqdm(frames, desc="Upscaling Frames (In-Memory)")]
                imageio.mimwrite(temp_video_path, upscaled_frames, fps=fps, quality=8)
            
            # Close resources before muxing
            if writer: writer.close(); writer = None
            if reader: reader.close(); reader = None

            final_temp_output = self._generate_output_path(video_path, f"upscaled_{model_key}", is_temp=True)
            if self.has_ffmpeg and self._has_audio_stream(video_path):
                print("Muxing audio into upscaled video...")
                mux_cmd = [self.ffmpeg_exe, "-y", "-i", str(temp_video_path), "-i", video_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", str(final_temp_output)]
                subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
                os.remove(temp_video_path)
            else: 
                shutil.move(temp_video_path, final_temp_output)
            
            return str(final_temp_output)
            
        except Exception as e: 
            print(f"Error during upscaling: {e}\n{traceback.format_exc()}")
            return video_path
            
        finally:
            if reader: reader.close()
            if writer: writer.close()
            
            self.esrgan_upscaler.unload_model(model_key)
            if enhance_face: self.esrgan_upscaler._unload_face_enhancer()
            if temp_video_path and os.path.exists(temp_video_path): os.remove(temp_video_path)
            
            gc.collect(); torch.cuda.empty_cache()

    def adjust_frames(self, video_path, fps_mode, speed_factor, use_streaming, progress=gr.Progress()):
        if not video_path: print("No input video for frame adjustment."); return None
        
        interpolation_factor = 1
        if "2x" in fps_mode: interpolation_factor = 2
        elif "4x" in fps_mode: interpolation_factor = 4
        should_interpolate = interpolation_factor > 1

        if not should_interpolate and speed_factor == 1.0:
            print("INFO: No frame interpolation or speed change requested. Skipping frame adjustment.")
            return video_path

        temp_video_path = None
        try:
            print(f"Adjusting frames: Mode={fps_mode}, Speed={speed_factor}x, Streaming: {use_streaming}")
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']
            output_fps = fps * interpolation_factor
            if use_streaming and speed_factor != 1.0:
                print("Note: Speed adjustment is ignored in RIFE streaming mode.")
                speed_factor = 1.0
            
            if use_streaming and should_interpolate:
                self.rife_handler._ensure_model_downloaded_and_loaded()
                temp_video_path = self._generate_output_path(video_path, "frames_temp", is_temp=True)
                writer = imageio.get_writer(temp_video_path, fps=output_fps, quality=8)
                frame_iterator = iter(reader)
                frame1 = next(frame_iterator, None)
                if frame1 is not None:
                    desc = f"Interpolating Frames ({interpolation_factor}x Streaming)"
                    for frame2 in progress.tqdm(frame_iterator, desc=desc):
                        writer.append_data(frame1)
                        middle = self.rife_handler.interpolate_between_frames(frame1, frame2)
                        if middle is not None: writer.append_data(middle)
                        frame1 = frame2
                    writer.append_data(frame1)
                writer.close()
            else:
                frames = [frame for frame in reader]
                processed_frames = frames
                if speed_factor != 1.0:
                    print(f"Adjusting speed by {speed_factor}x (in-memory)...")
                    new_len = int(len(frames) / speed_factor)
                    indices = np.linspace(0, len(frames) - 1, new_len).astype(int)
                    processed_frames = [frames[i] for i in indices]
                if should_interpolate and len(processed_frames) > 1:
                    self.rife_handler._ensure_model_downloaded_and_loaded()
                    num_passes = int(math.log2(interpolation_factor))
                    for p in range(num_passes):
                        print(f"INFO: Starting RIFE interpolation pass {p + 1}/{num_passes}...")
                        interpolated_this_pass = []
                        desc = f"RIFE Pass {p+1}/{num_passes}"
                        frame_iterator = progress.tqdm(range(len(processed_frames) - 1), desc=desc)
                        for i in frame_iterator:
                            interpolated_this_pass.append(processed_frames[i])
                            middle = self.rife_handler.interpolate_between_frames(processed_frames[i], processed_frames[i+1])
                            interpolated_this_pass.append(middle if middle is not None else processed_frames[i])
                        interpolated_this_pass.append(processed_frames[-1])
                        processed_frames = interpolated_this_pass
                temp_video_path = self._generate_output_path(video_path, "frames_temp", is_temp=True)
                imageio.mimwrite(temp_video_path, processed_frames, fps=output_fps, quality=8)
            reader.close()

            # --- Suffix and Final Path Generation ---
            suffix_parts = []
            if should_interpolate: suffix_parts.append(fps_mode.replace(' ', ''))
            if speed_factor != 1.0: suffix_parts.append(f"{speed_factor}x")
            suffix = f"frames_{'_'.join(suffix_parts)}"
            final_temp_output = self._generate_output_path(video_path, suffix, is_temp=True)

            # --- CORRECTED AUDIO MUXING LOGIC ---
            if self.has_ffmpeg and self._has_audio_stream(video_path):
                print("Muxing audio into processed video...")
                mux_cmd = [
                    self.ffmpeg_exe, "-y",
                    "-i", str(temp_video_path),
                    "-i", video_path,
                    "-c:v", "copy"
                ]
                
                # Conditionally apply the atempo filter ONLY if speed is changed
                if speed_factor != 1.0:
                    print(f"Applying atempo speed filter: {speed_factor}x")
                    audio_filters = [f"atempo={speed_factor}"]
                    if speed_factor > 2.0: audio_filters = [f"atempo=2.0,atempo={speed_factor/2.0}"]
                    if speed_factor < 0.5: audio_filters = [f"atempo=0.5,atempo={speed_factor/0.5}"]
                    mux_cmd.extend(["-filter:a", ",".join(audio_filters)])
                    mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) # Re-encode when filtering
                else:
                    # If just interpolating, copy the audio directly
                    print("Copying original audio without speed change.")
                    mux_cmd.extend(["-c:a", "copy"])

                mux_cmd.extend([
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", str(final_temp_output)
                ])
                
                subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
            else:
                # This block runs if there's no FFmpeg or no original audio
                shutil.move(temp_video_path, final_temp_output)

            return str(final_temp_output)
        except Exception as e:
            print(f"Error during frame adjustment: {e}\n{traceback.format_exc()}")
            return video_path
        finally:
            self.rife_handler.unload_model()
            if temp_video_path and os.path.exists(temp_video_path): os.remove(temp_video_path)
            gc.collect(); torch.cuda.empty_cache()

    def export_video(self, video_path, export_format, quality, max_width, output_name, progress=gr.Progress()):
        if not video_path: print("No input video to export."); return None
        if not self.has_ffmpeg: print("FFmpeg is required for export."); return None
        print(f"Exporting video to {export_format} with quality {quality} and max width {max_width}px.")
        try:
            ext = f".{export_format.lower()}"
            base_name = output_name if output_name and output_name.strip() else Path(video_path).stem
            suffix = f"exported_{max_width}w_{quality}q"
            # GIFs are always saved permanently, others respect autosave setting.
            is_temp_save = export_format != "GIF" and not self.autosave_enabled
            output_path = self._generate_output_path(base_name, suffix, ext=ext, is_temp=is_temp_save)
            if export_format == "GIF": print(f"INFO: GIF format selected. Output will be saved to permanent folder: {output_path}")

            ffmpeg_cmd = [self.ffmpeg_exe, "-y", "-i", video_path]
            if export_format == "MP4":
                crf = int(28 - (quality / 100) * 10)
                ffmpeg_cmd.extend(["-vf", f"scale='min({max_width},iw)':-2:flags=lanczos", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"])
            elif export_format == "WebM":
                crf = int(35 - (quality / 100) * 10)
                ffmpeg_cmd.extend(["-vf", f"scale='min({max_width},iw)':-2:flags=lanczos", "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-c:a", "libopus", "-b:a", "96k"])
            elif export_format == "GIF":
                progress(0.2, desc="Generating GIF palette (Pass 1/2)...")
                palette_path = self.temp_dir / "palette.png"
                palette_cmd = [self.ffmpeg_exe, "-y", "-i", video_path, "-vf", f"scale='min({max_width},iw)':-2:flags=lanczos,palettegen", str(palette_path)]
                subprocess.run(palette_cmd, check=True, capture_output=True, text=True)
                ffmpeg_cmd.extend(["-i", str(palette_path), "-filter_complex", f"[0:v]scale='min({max_width},iw)':-2:flags=lanczos[v];[v][1:v]paletteuse", "-an"])
            ffmpeg_cmd.append(str(output_path))
            progress(0.6, desc=f"Encoding {export_format}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: FFmpeg failed during export to {export_format}.\nCmd: {' '.join(e.cmd)}\nStderr: {e.stderr}"); return video_path
        except Exception as e: print(f"Error during export: {e}\n{traceback.format_exc()}"); return video_path

    def process_pipeline(self, input_path, operations, params, progress=gr.Progress()):
        """Processes a single video through a pipeline of operations."""
        current_video_path = input_path
        messages = [f"🚀 Starting pipeline for '{Path(input_path).name}'..."]
        execution_order = ["Upscale", "Frame Adjust", "Export"]
        for op_name in execution_order:
            if op_name in operations:
                messages.append(f"  -> Starting '{op_name}' step...")
                original_path = current_video_path
                if op_name == "Upscale": current_video_path = self.upscale_video(current_video_path, **params["upscale"], progress=progress)
                elif op_name == "Frame Adjust": current_video_path = self.adjust_frames(current_video_path, **params["frame_adjust"], progress=progress)
                elif op_name == "Export": current_video_path = self.export_video(current_video_path, **params["export"], progress=progress)
                if current_video_path == original_path:
                    messages.append(f"❌ Operation '{op_name}' failed. Aborting pipeline.")
                    return None, "\n".join(messages)
                else:
                    messages.append(f"  -> '{op_name}' step completed.")
        return current_video_path, "\n".join(messages)

    def process_batch(self, input_paths, operations, params, progress=gr.Progress()):
        """Processes a batch of videos through the pipeline."""
        total_videos, final_video_path = len(input_paths), None
        if total_videos == 0: return None, "No videos provided for batch processing."
        batch_messages = [f"🚀 Starting batch process for {total_videos} videos..."]
        batch_folder_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for i, video_path in enumerate(input_paths):
            progress(i / total_videos, desc=f"Processing video {i+1}/{total_videos}: {os.path.basename(video_path)}")
            batch_messages.append(f"\n--- Video {i+1}/{total_videos}: {os.path.basename(video_path)} ---")
            temp_result_path, messages = self.process_pipeline(video_path, operations, params, progress)
            batch_messages.append(messages)
            if temp_result_path:
                temp_path = Path(temp_result_path)
                final_path = self.output_dir / batch_folder_name / temp_path.name
                os.makedirs(final_path.parent, exist_ok=True)
                final_video_path = self._copy_to_permanent_storage(temp_result_path, final_path)
                batch_messages.append(f"✅ Batch result saved to: {final_path}")
            else: batch_messages.append(f"❌ Pipeline failed for {os.path.basename(video_path)}. Skipping.")
        batch_messages.append("\n--- ✅ Batch processing complete. ---")
        return final_video_path, "\n".join(batch_messages)
    
