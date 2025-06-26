"""
OpenAI API call for image processing
"""
import os
import base64
import json
import re
import cv2

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === GPT-4o bounding box helper and script ===
def detect_bounding_box(image_path, object_description):
    """
    Makes an API call to open AI to get the bounding box of an object given
    a description.

    Parameters: 
      image_path: path to the image file you want analyzed.
      object_description: Description of the object you want the VLM to find
        a bounding box for.

    Returns:
      The bounding box box coordinates.
    """
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the Base64 string
    base64_image = encode_image(image_path)

    # Load image to determine dimensions
    img = cv2.imread(image_path)
    
    h, w = img.shape[:2]
    print(f"Image dimensions: width={w}, height={h}")

    prompt_text = (
        f"The image is {w} pixels wide and {h} pixels tall. \
          In the image there is {object_description}. \
          Return the bounding box coordinates [x1, y1, x2, y2] of the \
          {object_description} in pixel coordinate space."
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

    print(f"OpenAI API response {response.output_text}")

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

    return coords
