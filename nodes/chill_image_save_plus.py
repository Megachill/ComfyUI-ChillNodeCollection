"""
ChillImageSavePlus - An enhanced image save node for ComfyUI
Supports multiple formats (PNG, JPEG, WebP, TIFF, BMP) with quality control and metadata options.
"""

import os
import re
import json
import numpy as np
from datetime import datetime
from PIL import Image, PngImagePlugin, TiffImagePlugin
try:
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False
    print("ChillImageSavePlus: piexif not installed — GPS EXIF for JPEG/WebP unavailable. Fix: pip install piexif")

import folder_paths
import comfy.utils


def _expand_date_macros(prefix):
    """Expand %date:FORMAT% and %time:FORMAT% macros in filename prefix.

    Supports Java SimpleDateFormat tokens: yyyy yy MM dd HH hh mm ss SSS
    """
    now = datetime.now()

    def _java_fmt_to_strftime(fmt):
        # Order matters: longer tokens first to avoid partial replacement
        fmt = fmt.replace("yyyy", "%Y")
        fmt = fmt.replace("yy", "%y")
        fmt = fmt.replace("MM", "%m")
        fmt = fmt.replace("dd", "%d")
        fmt = fmt.replace("HH", "%H")
        fmt = fmt.replace("hh", "%I")
        fmt = fmt.replace("mm", "%M")
        fmt = fmt.replace("ss", "%S")
        fmt = fmt.replace("SSS", "%f")
        return fmt

    def _replace(match):
        return now.strftime(_java_fmt_to_strftime(match.group(1)))

    prefix = re.sub(r'%date:([^%]+)%', _replace, prefix)
    prefix = re.sub(r'%time:([^%]+)%', _replace, prefix)
    return prefix


class ChillImageSavePlus:
    """
    Enhanced image save node supporting multiple formats with quality control
    and optional metadata stripping.
    """

    # Location presets with GPS coordinates (latitude, longitude, altitude in meters)
    LOCATION_PRESETS = {
        # Europe
        "Paris, France": (48.8566, 2.3522, 35),
        "London, UK": (51.5074, -0.1278, 11),
        "Berlin, Germany": (52.5200, 13.4050, 34),
        "Rome, Italy": (41.9028, 12.4964, 37),
        "Barcelona, Spain": (41.3851, 2.1734, 12),
        "Amsterdam, Netherlands": (52.3676, 4.9041, -2),
        "Vienna, Austria": (48.2082, 16.3738, 151),
        "Prague, Czech Republic": (50.0755, 14.4378, 177),
        "Athens, Greece": (37.9838, 23.7275, 70),
        "Stockholm, Sweden": (59.3293, 18.0686, 15),
        "Oslo, Norway": (59.9139, 10.7522, 7),
        # USA
        "New York City, USA": (40.7128, -74.0060, 10),
        "Los Angeles, USA": (34.0522, -118.2437, 89),
        "Chicago, USA": (41.8781, -87.6298, 181),
        "San Francisco, USA": (37.7749, -122.4194, 16),
        "Miami, USA": (25.7617, -80.1918, 2),
        "Las Vegas, USA": (36.1699, -115.1398, 610),
        "Seattle, USA": (47.6062, -122.3321, 56),
        "Denver, USA": (39.7392, -104.9903, 1609),
        # Canada
        "Toronto, Canada": (43.6532, -79.3832, 76),
        "Vancouver, Canada": (49.2827, -123.1207, 70),
        "Montreal, Canada": (45.5017, -73.5673, 35),
        "Calgary, Canada": (51.0447, -114.0719, 1045),
        # Russia
        "Moscow, Russia": (55.7558, 37.6173, 156),
        "St. Petersburg, Russia": (59.9311, 30.3609, 4),
        # Asia
        "Tokyo, Japan": (35.6762, 139.6503, 40),
        "Beijing, China": (39.9042, 116.4074, 44),
        "Singapore": (1.3521, 103.8198, 15),
        "Bangkok, Thailand": (13.7563, 100.5018, 4),
        "Seoul, South Korea": (37.5665, 126.9780, 38),
        "Mumbai, India": (19.0760, 72.8777, 14),
        "Dubai, UAE": (25.2048, 55.2708, 5),
        "Hong Kong": (22.3193, 114.1694, 7),
        # Latin America
        "Mexico City, Mexico": (19.4326, -99.1332, 2240),
        "Rio de Janeiro, Brazil": (-22.9068, -43.1729, 2),
        "Sao Paulo, Brazil": (-23.5505, -46.6333, 760),
        "Buenos Aires, Argentina": (-34.6037, -58.3816, 25),
        "Lima, Peru": (-12.0464, -77.0428, 154),
        "Bogota, Colombia": (4.7110, -74.0721, 2556),
        "Santiago, Chile": (-33.4489, -70.6693, 521),
        "Cusco, Peru": (-13.1631, -72.5450, 3399),
        # Oceania
        "Sydney, Australia": (-33.8688, 151.2093, 3),
        "Melbourne, Australia": (-37.8136, 144.9631, 31),
        # Manual entry option
        "Manual (use custom coords)": None,
    }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Chill"

    @classmethod
    def INPUT_TYPES(cls):
        # Get sorted location preset keys
        preset_keys = sorted([k for k in cls.LOCATION_PRESETS.keys()])

        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Chill"}),
                "format": (["png", "jpg", "webp", "tiff", "bmp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "strip_metadata": ("BOOLEAN", {"default": False}),
                "gps_enabled": ("BOOLEAN", {"default": False}),
                "gps_location_preset": (
                    preset_keys,
                    {"default": "Manual (use custom coords)"},
                ),
                "gps_latitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.000001},
                ),
                "gps_longitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.000001},
                ),
                "gps_altitude": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 10000.0, "step": 0.1},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    def save_images(
        self,
        images,
        filename_prefix,
        format,
        quality,
        strip_metadata,
        gps_enabled=False,
        gps_location_preset="Manual (use custom coords)",
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Save images in the specified format with quality control and optional metadata.

        Args:
            images: Tensor of shape (B, H, W, C) in range [0, 1]
            filename_prefix: Prefix for saved files
            format: Output format (png, jpg, webp, tiff, bmp)
            quality: Quality setting for lossy formats (1-100)
            strip_metadata: If True, strip all metadata
            prompt: Workflow prompt metadata
            extra_pnginfo: Additional PNG info metadata

        Returns:
            Dict with UI update information
        """
        # Expand date/time macros (e.g. %date:yyyy-MM-dd%) in filename prefix
        filename_prefix = _expand_date_macros(filename_prefix)

        # Clamp quality to valid range
        quality = max(1, min(100, quality))

        # Resolve GPS coordinates from preset if selected
        if gps_enabled and gps_location_preset != "Manual (use custom coords)":
            preset_coords = self.LOCATION_PRESETS.get(gps_location_preset)
            if preset_coords:
                gps_latitude, gps_longitude, gps_altitude = preset_coords

        # Map format strings to PIL format names
        format_mapping = {
            "png": "PNG",
            "jpg": "JPEG",
            "webp": "WEBP",
            "tiff": "TIFF",
            "bmp": "BMP",
        }
        pil_format = format_mapping.get(format, "PNG")

        # Determine if format supports quality parameter
        lossy_formats = {"JPEG", "WEBP"}
        supports_quality = pil_format in lossy_formats

        # Determine if format supports alpha channel
        alpha_formats = {"PNG", "TIFF", "WEBP"}
        supports_alpha = pil_format in alpha_formats

        results = []

        # Process each image in the batch
        for batch_idx, image in enumerate(images):
            # Convert tensor to numpy array (H, W, C)
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Handle alpha channel for formats that don't support it
            if not supports_alpha and img.mode in ("RGBA", "LA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if img.mode in ("RGBA", "LA"):
                    background.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = background
                else:
                    img = img.convert("RGB")

            # Get save path
            width, height = img.size
            full_output_folder, filename, counter, subfolder, _ = self._get_save_path(
                filename_prefix, self.output_dir, width, height
            )

            # Generate filename with correct extension
            file_name = f"{filename}_{counter:05d}.{format}"
            file_path = os.path.join(full_output_folder, file_name)

            # Ensure we don't overwrite existing files - increment counter if file exists
            while os.path.exists(file_path):
                counter += 1
                file_name = f"{filename}_{counter:05d}.{format}"
                file_path = os.path.join(full_output_folder, file_name)

            # Prepare save options
            save_kwargs = {}
            if supports_quality:
                save_kwargs["quality"] = quality
                if pil_format == "JPEG":
                    save_kwargs["optimize"] = True
                    save_kwargs["progressive"] = True
                elif pil_format == "WEBP":
                    save_kwargs["method"] = 6  # Best compression

            # Handle metadata
            jpeg_exif_dict = None  # built below; inserted into file after save

            if not strip_metadata:
                if pil_format == "PNG":
                    metadata = PngImagePlugin.PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for key, value in extra_pnginfo.items():
                            metadata.add_text(key, json.dumps(value))
                    if gps_enabled:
                        metadata.add_text("GPSLatitude", str(gps_latitude))
                        metadata.add_text("GPSLongitude", str(gps_longitude))
                        metadata.add_text("GPSAltitude", str(gps_altitude))
                    save_kwargs["pnginfo"] = metadata

                elif pil_format == "JPEG":
                    # PIL's exif= kwarg is silently ignored for freshly-created images;
                    # build the dict now and insert into the file via piexif after saving.
                    jpeg_exif_dict = self._build_exif_dict(
                        prompt, extra_pnginfo, gps_enabled,
                        gps_latitude, gps_longitude, gps_altitude,
                    )

                elif pil_format == "WEBP":
                    exif_bytes = self._create_exif_metadata(
                        prompt, extra_pnginfo, gps_enabled,
                        gps_latitude, gps_longitude, gps_altitude,
                    )
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes

                elif pil_format == "TIFF":
                    tiffinfo = self._create_tiff_metadata(
                        prompt, extra_pnginfo, gps_enabled,
                        gps_latitude, gps_longitude, gps_altitude,
                    )
                    if tiffinfo:
                        save_kwargs["tiffinfo"] = tiffinfo

            else:
                if img.mode in ("RGB", "RGBA", "L"):
                    img = Image.fromarray(np.array(img))
                else:
                    img = img.convert("RGB")

            # Save the image
            try:
                img.save(file_path, format=pil_format, **save_kwargs)
            except Exception as e:
                print(f"ChillImageSavePlus: Error saving {file_path}: {e}")
                raise

            # Post-save: inject EXIF into JPEG using piexif.insert()
            # (more reliable than PIL's exif= kwarg for new images)
            if jpeg_exif_dict is not None and _PIEXIF_AVAILABLE:
                try:
                    exif_bytes = piexif.dump(jpeg_exif_dict)
                    if len(exif_bytes) > 65500:
                        print("ChillImageSavePlus: EXIF too large, dropping workflow JSON but keeping GPS")
                        jpeg_exif_dict["Exif"] = {}
                        exif_bytes = piexif.dump(jpeg_exif_dict)
                    piexif.insert(exif_bytes, file_path)
                except Exception as e:
                    print(f"ChillImageSavePlus: Warning - could not insert JPEG EXIF: {e}")

            print(f"ChillImageSavePlus: Saved {file_path} (format={format}, quality={quality if supports_quality else 'N/A'})")

            results.append(
                {"filename": file_name, "subfolder": subfolder, "type": self.type}
            )

        return {"ui": {"images": results}}

    def _get_save_path(self, filename_prefix, output_dir, width, height):
        """
        Resolve output path for saving.

        ComfyUI >= 0.30.0 added a path-security check in get_save_image_path
        that uses os.path.commonpath, which incorrectly rejects valid subfolders
        when output_dir is a Windows drive root (e.g. Z:\\). We fall back to our
        own implementation that uses startswith, which handles drive roots correctly.
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
                        n = int(stem[len(filename) + 1:])
                        if n >= counter:
                            counter = n + 1
                    except ValueError:
                        pass
        except OSError:
            pass

        return full_output_folder, filename, counter, subfolder, filename_prefix

    def _decimal_to_dms(self, decimal_degrees):
        """
        Convert decimal degrees to EXIF DMS (degrees, minutes, seconds) format.

        Args:
            decimal_degrees: Decimal degrees value

        Returns:
            Tuple of 3 rationals: (degrees, minutes, seconds)
        """
        # Get absolute value for calculation
        abs_value = abs(decimal_degrees)

        # Extract degrees
        degrees = int(abs_value)

        # Extract minutes
        minutes_float = (abs_value - degrees) * 60
        minutes = int(minutes_float)

        # Extract seconds (with 2 decimal places precision)
        seconds = (minutes_float - minutes) * 60
        seconds_rational = (int(seconds * 100), 100)

        return ((degrees, 1), (minutes, 1), seconds_rational)

    def _build_exif_dict(
        self,
        prompt,
        extra_pnginfo,
        gps_enabled=False,
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
    ):
        """Return a piexif-compatible EXIF dict, or None if piexif unavailable."""
        if not _PIEXIF_AVAILABLE:
            return None
        try:
            zeroth_ifd = {
                piexif.ImageIFD.ImageDescription: b"ComfyUI Workflow",
                piexif.ImageIFD.Software: b"ComfyUI - ChillImageSavePlus",
            }
            exif_ifd = {}
            gps_ifd = {}

            metadata_dict = {}
            if prompt is not None:
                metadata_dict["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata_dict.update(extra_pnginfo)
            if metadata_dict:
                json_bytes = json.dumps(metadata_dict, separators=(",", ":")).encode("ascii", errors="replace")
                exif_ifd[piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + json_bytes

            if gps_enabled:
                if not (-90 <= gps_latitude <= 90):
                    raise ValueError(f"Latitude {gps_latitude} out of range [-90, 90]")
                if not (-180 <= gps_longitude <= 180):
                    raise ValueError(f"Longitude {gps_longitude} out of range [-180, 180]")
                gps_ifd = {
                    piexif.GPSIFD.GPSVersionID: b"\x02\x02\x00\x00",
                    piexif.GPSIFD.GPSLatitudeRef: b"N" if gps_latitude >= 0 else b"S",
                    piexif.GPSIFD.GPSLatitude: self._decimal_to_dms(gps_latitude),
                    piexif.GPSIFD.GPSLongitudeRef: b"E" if gps_longitude >= 0 else b"W",
                    piexif.GPSIFD.GPSLongitude: self._decimal_to_dms(gps_longitude),
                    piexif.GPSIFD.GPSAltitudeRef: b"\x00" if gps_altitude >= 0 else b"\x01",
                    piexif.GPSIFD.GPSAltitude: (int(abs(gps_altitude) * 10), 10),
                }

            return {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}}

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not build EXIF dict: {e}")
            return None

    def _create_exif_metadata(
        self,
        prompt,
        extra_pnginfo,
        gps_enabled=False,
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
    ):
        """Return piexif EXIF bytes for WebP (JPEG uses _build_exif_dict + piexif.insert instead)."""
        exif_dict = self._build_exif_dict(prompt, extra_pnginfo, gps_enabled, gps_latitude, gps_longitude, gps_altitude)
        if exif_dict is None:
            return None
        try:
            exif_bytes = piexif.dump(exif_dict)
            if len(exif_bytes) > 65500 and exif_dict["Exif"]:
                print("ChillImageSavePlus: EXIF too large, dropping workflow JSON but keeping GPS")
                exif_dict["Exif"] = {}
                exif_bytes = piexif.dump(exif_dict)
            return exif_bytes
        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create EXIF metadata: {e}")
            return None

    def _create_tiff_metadata(
        self,
        prompt,
        extra_pnginfo,
        gps_enabled=False,
        gps_latitude=0.0,
        gps_longitude=0.0,
        gps_altitude=0.0,
    ):
        """
        Create TIFF metadata dictionary.

        Args:
            prompt: Workflow prompt
            extra_pnginfo: Additional metadata
            gps_enabled: Whether to include GPS metadata
            gps_latitude: GPS latitude (-90 to 90)
            gps_longitude: GPS longitude (-180 to 180)
            gps_altitude: GPS altitude in meters

        Returns:
            Dict of TIFF tags or None
        """
        try:
            tiffinfo = {}

            metadata_dict = {}
            if prompt is not None:
                metadata_dict["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata_dict.update(extra_pnginfo)

            if metadata_dict:
                metadata_json = json.dumps(metadata_dict, separators=(",", ":"))

                # ImageDescription tag (270)
                tiffinfo[270] = metadata_json

            # Software tag (305)
            tiffinfo[305] = "ComfyUI - ChillImageSavePlus"

            # Add GPS metadata if enabled
            if gps_enabled:
                if not (-90 <= gps_latitude <= 90):
                    raise ValueError(f"Latitude {gps_latitude} out of range [-90, 90]")
                if not (-180 <= gps_longitude <= 180):
                    raise ValueError(
                        f"Longitude {gps_longitude} out of range [-180, 180]"
                    )

                # TIFF uses same GPS tags as EXIF via GDAL metadata
                tiffinfo[34853] = {  # GPSInfo tag
                    0: (2, 2, 0, 0),  # GPSVersionID
                    1: "N" if gps_latitude >= 0 else "S",  # GPSLatitudeRef
                    2: self._decimal_to_dms(gps_latitude),  # GPSLatitude
                    3: "E" if gps_longitude >= 0 else "W",  # GPSLongitudeRef
                    4: self._decimal_to_dms(gps_longitude),  # GPSLongitude
                    5: 0 if gps_altitude >= 0 else 1,  # GPSAltitudeRef
                    6: (int(abs(gps_altitude) * 10), 10),  # GPSAltitude
                }

            return tiffinfo if tiffinfo else None

        except Exception as e:
            print(f"ChillImageSavePlus: Warning - could not create TIFF metadata: {e}")

        return None


# Node registration
NODE_CLASS_MAPPINGS = {"ChillImageSavePlus": ChillImageSavePlus}

NODE_DISPLAY_NAME_MAPPINGS = {"ChillImageSavePlus": "Chill Image Save Plus"}
