import os
from PIL import Image
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateEntry,
    ImageFileCreateBatch,
    Region
)
from msrest.authentication import ApiKeyCredentials

# =========================================================
# INPUTS
# =========================================================

IMAGES_FOLDER = r"C:\Users\Donald Paterson\Documents\Donald_Paterson-AI_Engineer\Rosti\annotation_agent\output\plots_false\ROSTI-full_dataset\POSTOP1\images"
LABELS_FOLDER = r"C:\Users\Donald Paterson\Documents\Donald_Paterson-AI_Engineer\Rosti\annotation_agent\output\plots_false\ROSTI-full_dataset\POSTOP1\labels"
LABELS_TXT_PATH = r"C:\Users\Donald Paterson\Documents\Donald_Paterson-AI_Engineer\Rosti\annotation_agent\output\plots_false\ROSTI-full_dataset\POSTOP1\labels.txt"

TRAINING_KEY = "3okqTrDu3SwORxzSSc1S8NHgbCndDBaLTkLbYVa7hhWlau2oOrCjJQQJ99BEACGhslBXJ3w3AAAJACOGf91U"
ENDPOINT = "https://ncrpoc.cognitiveservices.azure.com/"

PROJECT_ID = "859258cf-0315-4a05-95ce-8973f8749005"

# =========================================================
# AUTHENTICATION
# =========================================================

credentials = ApiKeyCredentials(
    in_headers={"Training-key": TRAINING_KEY}
)

trainer = CustomVisionTrainingClient(
    ENDPOINT,
    credentials
)

# =========================================================
# LOAD LABELS
# =========================================================

def load_labels(labels_txt_path):
    with open(labels_txt_path, "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(LABELS_TXT_PATH)

# =========================================================
# CREATE / FETCH TAGS
# =========================================================

existing_tags = {
    tag.name: tag.id
    for tag in trainer.get_tags(PROJECT_ID)
}

tag_dict = {}

for label in labels:

    if label in existing_tags:
        tag_dict[label] = existing_tags[label]
        print(f"✅ Existing Tag: {label}")

    else:
        tag = trainer.create_tag(PROJECT_ID, label)
        tag_dict[label] = tag.id
        print(f"🆕 Created Tag: {label}")

# =========================================================
# IMAGE VALIDATION
# =========================================================

MIN_SIZE = 256
MAX_SIZE = 4096

def validate_image(img, image_name):

    width, height = img.size

    if width < MIN_SIZE or height < MIN_SIZE:
        print(f"❌ Image too small: {image_name} ({width}x{height})")
        return False

    if width > MAX_SIZE or height > MAX_SIZE:
        print(f"❌ Image too large: {image_name} ({width}x{height})")
        return False

    return True

# =========================================================
# UPLOAD FUNCTION
# =========================================================

def upload_images_one_by_one():

    image_files = os.listdir(IMAGES_FOLDER)

    uploaded_count = 0
    failed_count = 0

    for image_filename in image_files:

        image_path = os.path.join(IMAGES_FOLDER, image_filename)

        label_filename = os.path.splitext(image_filename)[0] + ".txt"

        label_path = os.path.join(LABELS_FOLDER, label_filename)

        # -------------------------------------------------
        # CHECK LABEL FILE
        # -------------------------------------------------

        if not os.path.exists(label_path):
            print(f"⚠️ Missing label file: {label_filename}")
            failed_count += 1
            continue

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        try:

            with Image.open(image_path) as img:

                img = img.convert("RGB")

                if not validate_image(img, image_filename):
                    failed_count += 1
                    continue

                img_width, img_height = img.size

        except Exception as e:
            print(f"❌ Failed opening image: {image_filename}")
            print(e)
            failed_count += 1
            continue

        # -------------------------------------------------
        # READ YOLO LABELS
        # -------------------------------------------------

        regions = []

        try:

            with open(label_path, "r") as f:

                for line in f.readlines():

                    parts = line.strip().split()

                    if len(parts) != 5:
                        print(f"⚠️ Invalid label format in {label_filename}")
                        continue

                    class_id = int(parts[0])

                    if class_id >= len(labels):
                        print(f"⚠️ Invalid class id in {label_filename}")
                        continue

                    x_center, y_center, width, height = map(
                        float,
                        parts[1:]
                    )

                    left = x_center - (width / 2)
                    top = y_center - (height / 2)

                    region = Region(
                        tag_id=tag_dict[labels[class_id]],
                        left=left,
                        top=top,
                        width=width,
                        height=height
                    )

                    regions.append(region)

        except Exception as e:
            print(f"❌ Error reading label file: {label_filename}")
            print(e)
            failed_count += 1
            continue

        # -------------------------------------------------
        # READ IMAGE BYTES
        # -------------------------------------------------

        try:

            with open(image_path, "rb") as image_file:

                image_contents = image_file.read()

        except Exception as e:
            print(f"❌ Failed reading image bytes: {image_filename}")
            print(e)
            failed_count += 1
            continue

        # -------------------------------------------------
        # CREATE ENTRY
        # -------------------------------------------------

        entry = ImageFileCreateEntry(
            name=image_filename,
            contents=image_contents,
            regions=regions
        )

        batch = ImageFileCreateBatch(
            images=[entry]
        )

        # -------------------------------------------------
        # UPLOAD
        # -------------------------------------------------

        try:

            result = trainer.create_images_from_files(
                PROJECT_ID,
                batch
            )

            if result.is_batch_successful:

                uploaded_count += 1

                print(
                    f"✅ Uploaded: {image_filename} "
                    f"({uploaded_count}/{len(image_files)})"
                )

            else:

                failed_count += 1

                print(f"❌ Upload Failed: {image_filename}")

                for img in result.images:
                    print(f"   Status: {img.status}")

        except Exception as e:

            failed_count += 1

            print(f"❌ Exception during upload: {image_filename}")
            print(e)

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n===================================")
    print("UPLOAD FINISHED")
    print("===================================")

    print(f"✅ Uploaded : {uploaded_count}")
    print(f"❌ Failed   : {failed_count}")
    print(f"📦 Total    : {len(image_files)}")

# =========================================================
# RUN
# =========================================================

upload_images_one_by_one()