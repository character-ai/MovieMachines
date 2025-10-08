import tempfile
from typing import Optional
import os

import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
from scipy.io import wavfile


def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.
    """

    assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

    if audio_numpy is not None:
        assert isinstance(audio_numpy, np.ndarray), "audio_numpy must be a numpy array"
        assert np.abs(audio_numpy).max() <= 1.0, "audio_numpy values must be in range [-1, 1]"

    video_numpy = video_numpy.transpose(1, 2, 3, 0)

    if video_numpy.max() <= 1.0:
        video_numpy = np.clip(video_numpy, -1, 1)
        video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    frames = list(video_numpy)
    clip = ImageSequenceClip(frames, fps=fps)
    temp_audio_path = None
    final_clip = clip

    try:
        if audio_numpy is not None:
            # Create a temp file but don't delete it on close, so we control the lifecycle
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = temp_audio_file.name
            
            # Write the audio data to the path
            wavfile.write(
                temp_audio_path,
                sample_rate,
                (audio_numpy * 32767).astype(np.int16),
            )
            # Load the audio into a clip
            audio_clip = AudioFileClip(temp_audio_path)
            final_clip = clip.set_audio(audio_clip)

        # Write the final video. This is when moviepy locks the temp audio file.
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None
        )

    finally:
        
        # Close the clip to release all file handles.
        final_clip.close()
        
        # Now that the file is released, we can safely delete our temp audio file
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return output_path