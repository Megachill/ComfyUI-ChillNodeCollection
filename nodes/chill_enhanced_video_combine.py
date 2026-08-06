"""
ChillEnhancedVideoCombine - Combine an IMAGE batch into an encoded video via FFmpeg.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime

import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    import comfy.utils
except ImportError:
    comfy = None


_ENCODER_CANDIDATES = {
    "H.264": ("h264_nvenc", "h264_qsv", "h264_amf", "libx264"),
    "H.265": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"),
    "VP9": ("libvpx-vp9",),
    "AV1": ("av1_nvenc", "libsvtav1", "libaom-av1"),
}
_CONTAINER_EXT = {"mp4": ".mp4", "webm": ".webm", "mkv": ".mkv"}
_CONTAINER_MIME = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska"}
_CODEC_CONTAINERS = {
    "H.264": ("mp4", "mkv"),
    "H.265": ("mp4", "mkv"),
    "VP9": ("webm", "mkv"),
    "AV1": ("webm", "mkv"),
}
_AUDIO_CODEC_OPTIONS = ["Auto", "AAC", "Opus", "MP3"]
_AUDIO_ENCODERS = {"AAC": "aac", "Opus": "libopus", "MP3": "libmp3lame"}
_AUDIO_BITRATE_OPTIONS = ["64k", "96k", "128k", "160k", "192k", "256k", "320k"]


def _expand_filename_macros(prefix):
    """Expand %date:FORMAT% / %time:FORMAT% tokens using Java SimpleDateFormat-style patterns."""
    now = datetime.now()

    def _to_strftime(fmt):
        fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y")
        fmt = fmt.replace("MM", "%m").replace("dd", "%d")
        fmt = fmt.replace("HH", "%H").replace("hh", "%I")
        fmt = fmt.replace("mm", "%M").replace("ss", "%S")
        return fmt

    def _replace(match):
        return now.strftime(_to_strftime(match.group(1)))

    prefix = re.sub(r"%date:([^%]+)%", _replace, prefix)
    prefix = re.sub(r"%time:([^%]+)%", _replace, prefix)
    return prefix


def _find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _available_encoders(ffmpeg):
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return set()
    return {line.split()[1] for line in result.stdout.splitlines() if line.startswith(" V")}


def _pick_encoder(ffmpeg, codec):
    available = _available_encoders(ffmpeg)
    for candidate in _ENCODER_CANDIDATES[codec]:
        if candidate in available:
            return candidate
    # Probe may fail (e.g. ffmpeg missing -encoders support); fall back to software.
    return _ENCODER_CANDIDATES[codec][-1]


def _audio_encoder(audio_codec, container):
    if audio_codec == "Auto":
        return "libopus" if container == "webm" else "aac"
    return _AUDIO_ENCODERS[audio_codec]


def _resolve_save_path(filename_prefix, output_dir, width, height):
    """
    Resolve the output folder/filename/counter for a save.

    ComfyUI >= 0.30.0 added a path-security check in get_save_image_path that
    uses os.path.commonpath, which incorrectly rejects valid subfolders when
    output_dir is a Windows drive root (e.g. Z:\\). Fall back to our own
    startswith-based resolution, which handles drive roots correctly.
    """
    try:
        return folder_paths.get_save_image_path(filename_prefix, output_dir, width, height)
    except Exception as e:
        if "outside the output folder" not in str(e):
            raise

    norm_prefix = os.path.normpath(filename_prefix)
    subfolder = os.path.dirname(norm_prefix)
    filename = os.path.basename(norm_prefix)
    full_output_folder = os.path.normpath(os.path.join(output_dir, subfolder))
    os.makedirs(full_output_folder, exist_ok=True)

    counter = 1
    try:
        prefix_lower = os.path.normcase(filename) + "_"
        for f in os.listdir(full_output_folder):
            stem = os.path.splitext(f)[0]
            if os.path.normcase(stem).startswith(prefix_lower):
                try:
                    n = int(stem[len(filename) + 1:].split("-")[0])
                    if n >= counter:
                        counter = n + 1
                except ValueError:
                    pass
    except OSError:
        pass

    return full_output_folder, filename, counter, subfolder, filename_prefix


def _write_audio_wav(audio):
    """Write a ComfyUI AUDIO value to a temp 16-bit PCM WAV file. Returns (path, duration_seconds)."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if waveform.ndim == 3:
        waveform = waveform[0]  # drop batch dim
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32).clamp_(-1, 1)
    channels = waveform.shape[0]
    interleaved = waveform.transpose(0, 1).numpy()
    pcm16 = np.round(interleaved * 32767.0).astype(np.int16)

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    with wave.open(handle.name, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    duration = pcm16.shape[0] / sample_rate
    return handle.name, duration


def _pingpong_frames(images, pingpong):
    if not pingpong or len(images) < 3:
        return images
    return torch.cat((images, images[1:-1].flip(0)), dim=0)


def _frames_to_rgb_bytes(frames):
    rgb = frames[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
    return torch.round(rgb * 255).to(torch.uint8).numpy().tobytes()


def _export_edge_frames(images, output_path, save_first_frame, save_last_frame, pingpong):
    """Write the first and/or last source frame as a PNG beside the encoded video."""
    exports = []
    if not save_first_frame and not save_last_frame:
        return exports

    base_path = os.path.splitext(output_path)[0]

    def _write_png(frame, suffix):
        rgb = frame[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
        pixels = torch.round(rgb * 255).to(torch.uint8).numpy()
        path = f"{base_path}_{suffix}.png"
        Image.fromarray(pixels, mode="RGB").save(path, "PNG")
        exports.append(path)

    if save_first_frame:
        _write_png(images[0], "first")
    if save_last_frame:
        last_index = 1 if (pingpong and len(images) >= 3) else -1
        _write_png(images[last_index], "last")

    return exports


class ChillEnhancedVideoCombine:
    DESCRIPTION = (
        "Combines an IMAGE batch into an encoded video via FFmpeg, with GPU encoder "
        "auto-detection, optional audio mux, ping-pong loops, and %date% filename macros."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "codec": (["H.264", "H.265", "VP9", "AV1"], {"default": "H.264"}),
                "container": (["mp4", "webm", "mkv"], {"default": "mp4"}),
                "quality": (
                    "INT",
                    {"default": 20, "min": 0, "max": 51, "description": "FFmpeg CRF. Lower = higher quality/larger file."},
                ),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_metadata": (
                    "BOOLEAN",
                    {"default": True, "description": "Write the ComfyUI workflow as a sidecar .json file next to the video."},
                ),
                "filename_prefix": ("STRING", {"default": "video/%date:yyyy-MM-dd%/Chill_%date:hhmmss%"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False, "description": "Return the encoded frame sequence for downstream processing."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "crop_to_audio": (
                    "BOOLEAN",
                    {"default": False, "description": "End the video at the audio's duration."},
                ),
                "audio_codec": (_AUDIO_CODEC_OPTIONS, {"default": "Auto"}),
                "audio_bitrate": (_AUDIO_BITRATE_OPTIONS, {"default": "192k"}),
                "save_first_frame": ("BOOLEAN", {"default": False, "description": "Write the first frame as a PNG beside the encoded video."}),
                "save_last_frame": ("BOOLEAN", {"default": False, "description": "Write the last frame as a PNG beside the encoded video."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "filename")
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "Chill/Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def combine(
        self,
        images,
        frame_rate,
        codec,
        container,
        quality,
        pingpong,
        save_metadata,
        filename_prefix,
        save_output,
        pass_frames,
        audio=None,
        crop_to_audio=False,
        audio_codec="Auto",
        audio_bitrate="192k",
        save_first_frame=False,
        save_last_frame=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an IMAGE batch shaped [frames, height, width, channels].")

        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg was not found. Install FFmpeg or the imageio-ffmpeg Python package.")

        if container not in _CODEC_CONTAINERS[codec]:
            fallback = _CODEC_CONTAINERS[codec][0]
            print(f"ChillEnhancedVideoCombine: {codec} does not support .{container}, using .{fallback} instead")
            container = fallback

        filename_prefix = _expand_filename_macros(filename_prefix)
        height, width = images.shape[1:3]
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        output_type = "output" if save_output else "temp"

        full_output_folder, filename, counter, subfolder, _ = _resolve_save_path(
            filename_prefix, output_dir, width, height
        )

        extension = _CONTAINER_EXT[container]
        file_name = f"{filename}_{counter:05d}{extension}"
        output_path = os.path.join(full_output_folder, file_name)
        while os.path.exists(output_path):
            counter += 1
            file_name = f"{filename}_{counter:05d}{extension}"
            output_path = os.path.join(full_output_folder, file_name)

        encoder = _pick_encoder(ffmpeg, codec)

        audio_path = None
        audio_duration = None
        if audio is not None:
            audio_path, audio_duration = _write_audio_wav(audio)

        command = [ffmpeg, "-y", "-v", "error"]
        command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "-"]
        if audio_path:
            command += ["-i", audio_path]
            command += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", _audio_encoder(audio_codec, container), "-b:a", audio_bitrate]
            if crop_to_audio and audio_duration:
                command += ["-t", f"{audio_duration:.6f}"]
        command += ["-c:v", encoder, "-crf", str(quality), "-pix_fmt", "yuv420p"]
        if container == "mp4":
            command += ["-movflags", "+faststart"]
        command += [output_path]

        pingponged = _pingpong_frames(images, pingpong)
        frame_bytes = _frames_to_rgb_bytes(pingponged)
        num_encoded_frames = len(pingponged)
        progress_bar = comfy.utils.ProgressBar(num_encoded_frames) if comfy is not None else None

        try:
            process = subprocess.run(command, input=frame_bytes, capture_output=True, timeout=3600)
        finally:
            if audio_path:
                os.unlink(audio_path)

        if progress_bar is not None:
            progress_bar.update_absolute(num_encoded_frames)

        if process.returncode != 0:
            error = process.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed ({encoder}): {error}")

        if save_metadata and (prompt is not None or extra_pnginfo):
            metadata = {}
            if prompt is not None:
                metadata["prompt"] = prompt
            if extra_pnginfo:
                metadata.update(extra_pnginfo)
            sidecar_path = os.path.splitext(output_path)[0] + ".json"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

        frame_exports = _export_edge_frames(images, output_path, save_first_frame, save_last_frame, pingpong)

        print(f"ChillEnhancedVideoCombine: Saved {output_path} ({codec} via {encoder}, {container})")
        for path in frame_exports:
            print(f"ChillEnhancedVideoCombine: Frame export {path}")

        assets = [
            {
                "filename": file_name,
                "subfolder": subfolder,
                "type": output_type,
                "format": _CONTAINER_MIME[container],
                "width": width,
                "height": height,
                "fps": frame_rate,
            }
        ]
        assets.extend(
            {"filename": os.path.basename(path), "subfolder": subfolder, "type": output_type, "format": "image/png"}
            for path in frame_exports
        )

        output_frames = pingponged if pass_frames else images[:0]
        return {"ui": {"gifs": assets}, "result": (output_frames, output_path)}


# Node registration
NODE_CLASS_MAPPINGS = {"ChillEnhancedVideoCombine": ChillEnhancedVideoCombine}

NODE_DISPLAY_NAME_MAPPINGS = {"ChillEnhancedVideoCombine": "Chill Enhanced Video Combine"}
