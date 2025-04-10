from PIL import Image
import numpy as np
import cv2
import apriltag

# Create tag
tag_id = 0
tag_family = 'tag36h11'
tag_size_mm = 146
dpi = 300  # High quality print

# Create AprilTag image
generator = apriltag._get_demo_tag_family(tag_family)
tag_image = generator.generate(tag_id)

# Convert to PIL
img = Image.fromarray(tag_image * 255).convert('L')

# Calculate pixel size for 146mm at 300 DPI
pixels = int((tag_size_mm / 25.4) * dpi)
img = img.resize((pixels, pixels), Image.NEAREST)

# Save
img.save(f"apriltag_{tag_id}_{tag_size_mm}mm.png", dpi=(dpi, dpi))
