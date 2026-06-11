import os
import json
import requests
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
from tqdm import tqdm
import yaml
import random

# Azure Custom Vision Configuration
# ENDPOINT = "https://roshvisionpoc.cognitiveservices.azure.com/"
# TRAINING_KEY = "cde01f9795074f0b9ed97df8643306e7"
PROJECT_ID = "859258cf-0315-4a05-95ce-8973f8749005"

ENDPOINT = "https://ncrpoc.cognitiveservices.azure.com/"
TRAINING_KEY = "3okqTrDu3SwORxzSSc1S8NHgbCndDBaLTkLbYVa7hhWlau2oOrCjJQQJ99BEACGhslBXJ3w3AAAJACOGf91U"


# Azure Custom Vision Configuration
# ENDPOINT = "https://nokitatratiningtepm.cognitiveservices.azure.com/"
# TRAINING_KEY = "62bj0inj3CxMzWDnQdcI618xx249H2geogsWQipisNAi359X6Y5iJQQJ99BIACYeBjFXJ3w3AAAJACOGTK3R"

# autolabeler
# ENDPOINT = "https://nokiatraining.cognitiveservices.azure.com/"
# TRAINING_KEY = "7hgLcZMlUZC4rGq0k6pDNkoHRVPLfCToNCTE27y0tpOoaXr6pKZTJQQJ99BGACGhslBXJ3w3AAAJACOGCfy6"

# ENDPOINT = "https://nokitatratiningtepm.cognitiveservices.azure.com/"
# TRAINING_KEY = "62bj0inj3CxMzWDnQdcI618xx249H2geogsWQipisNAi359X6Y5iJQQJ99BIACYeBjFXJ3w3AAAJACOGTK3R"

# ENDPOINT = "https://roshai-autoai-inspection-pov.cognitiveservices.azure.com/"
# TRAINING_KEY = "09c56f32ad0048c1afb2f8ea6f59b1e5"


# ENDPOINT = "https://roshai-autoai-inspection-pov.cognitiveservices.azure.com/"
# TRAINING_KEY = "09c56f32ad0048c1afb2f8ea6f59b1e5"

# ENDPOINT = "https://nokiatraining.cognitiveservices.azure.com/"
# TRAINING_KEY = "7hgLcZMlUZC4rGq0k6pDNkoHRVPLfCToNCTE27y0tpOoaXr6pKZTJQQJ99BGACGhslBXJ3w3AAAJACOGCfy6"

credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

output_dir = "Rosti-top_1-validated"  # Change this to your desired output directory
print(output_dir)
# output_dir = "quasarall"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

print("Fetching class names...")
tags = trainer.get_tags(PROJECT_ID)
class_names = [tag.name for tag in tags]
labels_file = os.path.join(output_dir, "labels.txt")
with open(labels_file, "w") as f:
    f.write("\n".join(class_names))
print(f"Classes: {class_names}")
desired_count = 5000 # Change this to how many images you want
skip = 0
batch_size = 256  # Max allowed by API
tagged_images = []

print(f"Fetching up to {desired_count} tagged images...")

while len(tagged_images) < desired_count:
    remaining = desired_count - len(tagged_images)
    take = min(batch_size, remaining)
    
    batch = trainer.get_tagged_images(PROJECT_ID, skip=skip, take=take)
    if not batch:
        print("No more tagged images available.")
        break

    tagged_images.extend(batch)
    skip += take

print(f"Total images fetched: {len(tagged_images)}")
# Split images into training and validation sets
# random.shuffle(tagged_images)
# train_size = int(0.8 * len(tagged_images))  # 80% for training
# train_images = tagged_images[:train_size]
# val_images = tagged_images[train_size:]

# Save images and annotations
for image in tqdm(tagged_images, desc="Downloading images"):
    image_id = image.id
    image_url = image.original_image_uri
    image_name = f"{image_id}.jpg"
    label_name = f"{image_id}.txt"

    # Download image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        with open(os.path.join(images_dir, image_name), "wb") as img_file:
            img_file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {image_name}: {e}")

    # Save annotation (YOLO format)
    annotations = []
    for region in image.regions:
        tag_name = region.tag_name
        class_id = class_names.index(tag_name)
        left = region.left
        top = region.top
        width = region.width
        height = region.height

        x_center = left + width / 2
        y_center = top + height / 2

        annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    if annotations:
        with open(os.path.join(labels_dir, label_name), "w") as label_file:
            label_file.write("\n".join(annotations))

print("Dataset preparation completed!")

# Create data.yaml for YOLO
data_yaml_content = {
    "train": os.path.join(images_dir, "train"),
    "val": os.path.join(images_dir, "val"),
    "names": class_names
}

with open("data.yaml", "w") as yaml_file:
    yaml.dump(data_yaml_content, yaml_file, default_flow_style=False)

print("data.yaml file generated!")
