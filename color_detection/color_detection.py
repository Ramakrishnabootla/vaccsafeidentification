import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------
# LOAD YOLOv8 SEGMENTATION MODEL
# ---------------------------------
model = YOLO("yolov8s-seg.pt")

# ---------------------------------
# FINAL COLOR RANGES
# ONLY:
# white
# black
# brown
# ---------------------------------
COLOR_RANGES = {

    # BLACK DOGS
    "black": ([0, 0, 0], [180, 255, 55]),

    # WHITE + CREAM + LIGHT GOLDEN
    "white": ([0, 0, 160], [180, 80, 255]),

    # BROWN + DARK GOLDEN
    "brown": ([5, 60, 20], [30, 255, 200]),
}

# ---------------------------------
# ROBUST COLOR DETECTION
# ---------------------------------
def detect_color(image_rgb):

    # ---------------------------------
    # REMOVE BLACK BACKGROUND
    # ---------------------------------
    pixels = image_rgb.reshape(-1, 3)

    pixels = pixels[np.sum(pixels, axis=1) > 60]

    if len(pixels) == 0:
        return "unknown"

    # ---------------------------------
    # RGB -> HSV
    # ---------------------------------
    pixels = np.uint8(pixels).reshape(-1, 1, 3)

    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)

    hsv = hsv.reshape(-1, 3)

    # remove shadow/noise pixels
    hsv = hsv[hsv[:, 2] > 25]

    if len(hsv) == 0:
        return "unknown"

    # ---------------------------------
    # BRIGHTNESS ANALYSIS
    # ---------------------------------
    brightness = np.mean(hsv[:, 2])

    saturation = np.mean(hsv[:, 1])

    # ---------------------------------
    # STRONG BLACK DETECTION
    # ---------------------------------
    if brightness < 85:
        return "black"

    # ---------------------------------
    # WHITE DETECTION
    # ---------------------------------
    if brightness > 170 and saturation < 80:
        return "white"

    # ---------------------------------
    # BROWN DETECTION
    # ---------------------------------
    return "brown"


# ---------------------------------
# IMAGE PATH
# ---------------------------------
image_path = "D:/1csm/Interns/Hexart/vacc_safe/Dogs_photos/Dog1.jpg"

# ---------------------------------
# READ IMAGE
# ---------------------------------
image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

# ---------------------------------
# YOLO DETECTION
# ---------------------------------
results = model(image)

for result in results:

    if result.masks is None:
        continue

    masks = result.masks.xy
    classes = result.boxes.cls

    for i, mask in enumerate(masks):

        class_id = int(classes[i])

        # ONLY DOG
        if model.names[class_id] != "dog":
            continue

        # ---------------------------------
        # CREATE SEGMENTATION MASK
        # ---------------------------------
        polygon = np.array(mask, dtype=np.int32)

        binary_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        cv2.fillPoly(
            binary_mask,
            [polygon],
            255
        )

        # ---------------------------------
        # EXTRACT ONLY DOG
        # ---------------------------------
        segmented = cv2.bitwise_and(
            image,
            image,
            mask=binary_mask
        )

        # ---------------------------------
        # GET BOUNDING BOX
        # ---------------------------------
        x, y, w, h = cv2.boundingRect(polygon)

        dog_crop = segmented[y:y+h, x:x+w]

        if dog_crop.size == 0:
            continue

        # ---------------------------------
        # NORMALIZE SIZE
        # ---------------------------------
        dog_crop = cv2.resize(
            dog_crop,
            (256, 256)
        )

        # ---------------------------------
        # PREPROCESS
        # ---------------------------------
        dog_rgb = cv2.cvtColor(
            dog_crop,
            cv2.COLOR_BGR2RGB
        )

        # blur improves fur consistency
        dog_rgb = cv2.GaussianBlur(
            dog_rgb,
            (5, 5),
            0
        )

        # ---------------------------------
        # DETECT COLOR
        # ---------------------------------
        detected_color = detect_color(dog_rgb)

        # FINAL OUTPUT
        print(f"Color Name : {detected_color}")