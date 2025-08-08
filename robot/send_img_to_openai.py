import base64
from openai import OpenAI
import cv2
import json
import re

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "original_image.png"

# Getting the Base64 string
base64_image = encode_image(image_path)

# Load image to determine dimensions
img = cv2.imread(image_path)
h, w = img.shape[:2]
print(f"Image dimensions: width={w}, height={h}")


prompt_text = (
    f"The image is {w} pixels wide and {h} pixels tall. "
    "Return the bounding box coordinates [x1, y1, x2, y2] of the vertical edge of the wooden partition or panel in the foreground of the image, in pixel coordinate space."
)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "system",
         "content": (
            "You are an assistant that only returns JSON. "
            "Given an image embed, you MUST output exactly "
            "one JSON array [x1,y1,x2,y2], nothing else."
        )},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt_text
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)

# Parse JSON bounding box coordinates
content = response.output_text.strip()
# Extract JSON array from descriptive text
match = re.search(r'\[.*?\]', content)
if not match:
    raise RuntimeError(f"Could not find bounding box JSON in response: {content}")
coords = json.loads(match.group(0))
x1, y1, x2, y2 = coords

# Load original image, draw box, and save annotated image
img = cv2.imread(image_path)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
annotated_path = "annotated_image.png"
cv2.imwrite(annotated_path, img)
print(f"Saved annotated image to {annotated_path}")