import os
import matplotlib.pyplot as plt

# Folder containing images
image_folder = "images_filtered"
output_txt_file = "bounding_boxes.txt"

# Get all image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.bmp'))])

# Open the TXT file for writing
with open(output_txt_file, "w") as file:
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)

        # Load image
        image = plt.imread(image_path)

        # Display image
        plt.imshow(image)
        plt.title(f"Select bounding box for {image_name}")

        # Get two user clicks
        points = plt.ginput(2, timeout=0)
        plt.close()

        if len(points) != 2:
            print(f"Skipping {image_name} (no bounding box selected)")
            continue

        # Convert to integer coordinates
        (x1, y1), (x2, y2) = points
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(f"Bounding Box for {image_name}: ({x1}, {y1}) to ({x2}, {y2})")

        # Write bounding box to file in the format: filename x1 y1 x2 y2
        file.write(f"{image_name} {x1} {y1} {x2} {y2}\n")

print(f"Bounding boxes saved to {output_txt_file}")
