
# channel to image mode mapping
CHANNEL_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def convert_image_to_fn(img_type, image):
    if img_type is None or image.mode == img_type:
        return image

    return image.convert(img_type)