"""
Chill Node Collection - ComfyUI Custom Nodes Package
"""

from .chill_image_save_plus import ChillImageSavePlus
from .chill_image_save_plus import NODE_CLASS_MAPPINGS as _IMAGE_SAVE_MAPPINGS
from .chill_image_save_plus import NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_SAVE_DISPLAY_MAPPINGS
from .chill_enhanced_video_combine import ChillEnhancedVideoCombine
from .chill_enhanced_video_combine import NODE_CLASS_MAPPINGS as _VIDEO_COMBINE_MAPPINGS
from .chill_enhanced_video_combine import NODE_DISPLAY_NAME_MAPPINGS as _VIDEO_COMBINE_DISPLAY_MAPPINGS

NODE_CLASS_MAPPINGS = {**_IMAGE_SAVE_MAPPINGS, **_VIDEO_COMBINE_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_IMAGE_SAVE_DISPLAY_MAPPINGS, **_VIDEO_COMBINE_DISPLAY_MAPPINGS}

__all__ = [
    "ChillImageSavePlus",
    "ChillEnhancedVideoCombine",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
