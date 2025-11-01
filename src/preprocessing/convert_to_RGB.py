from typing import Any

class ConvertToRGB:
    """
    This class checks and coverts the image to RGB mode if needed.
    """
    def __call__(self, image) -> Any:
        if image.mode != "RGB":
            image.convert("RGB")
        return image