# Chill Node Collection for ComfyUI

A collection of enhanced utility nodes for ComfyUI.

## Nodes

### Chill Image Save Plus

An enhanced image save node that extends the built-in `SaveImage` node with additional format support, quality control for lossy formats, GPS EXIF metadata, and `%date%`/`%time%` macros in the filename prefix.

- **Multiple Format Support**: PNG, JPEG, WebP, TIFF, BMP
- **Quality Control**: Adjustable quality (1-100) for lossy formats (JPEG, WebP)
- **GPS EXIF Metadata**: Embed GPS coordinates (with location presets or manual entry) into JPEG/WebP/TIFF
- **Date Macros**: `%date:yyyy-MM-dd%`, `%time:HHmmss%`, etc. in `filename_prefix`
- **Metadata Options**: Embed or strip workflow/prompt metadata
- **Batch Processing**: Handles batched images with automatic counter incrementing

### Chill Enhanced Video Combine

Combines an `IMAGE` batch into an encoded video via FFmpeg.

- **Codec/Container Selection**: H.264, H.265, VP9, AV1 with automatic GPU encoder detection (NVENC/QSV/AMF) and software fallback
- **Audio Mux**: Optionally mux a connected `AUDIO` input, with an option to crop the video to the audio's duration
- **Ping-Pong Loops**: Append interior frames in reverse for seamless forward/reverse playback
- **Date Macros**: Same `%date%`/`%time%` support as Chill Image Save Plus
- **Workflow Metadata**: Optionally writes the ComfyUI workflow as a sidecar `.json` file next to the video

Requires FFmpeg on `PATH`, or the `imageio-ffmpeg` Python package (included in requirements.txt).

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:Megachill/ComfyUI-ChillNodeCollection.git
```

### Method 2: Manual Download

1. Download this repository as a ZIP file
2. Extract it to `ComfyUI/custom_nodes/`
3. Rename the folder to `ComfyUI-ChillNodeCollection`

### Dependencies

```bash
pip install -r requirements.txt
```

**ComfyUI Portable (Windows):**

```bash
../../../python_embeded/python.exe -m pip install -r requirements.txt
```

Run this command from within the `ComfyUI/custom_nodes/ComfyUI-ChillNodeCollection/` folder.

## Usage

Both nodes appear in the node library under the **"Chill"** category (search for "Chill Image Save Plus" or "Chill Enhanced Video Combine").

## License

MIT
