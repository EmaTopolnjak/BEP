import os
import json
from PIL import Image, ImageOps



def rotate_image(image, angle):
    """ Rotate the image by the given angle and return the rotated image. The background 
    color is set to (245, 245, 245). The image is centered on a square canvas.

    Parameters:
    image (PIL.Image): The image to be rotated.
    angle (float): The angle to rotate the image.

    Returns:
    final_image (PIL.Image): The rotated image. """

    rotated = image.rotate(-angle, expand=True)

    # Create white square canvas
    w, h = rotated.size
    max_dim = max(w, h)
    square_img = Image.new("RGBA", (max_dim, max_dim), (245, 245, 245, 255)) # background color is (245, 245, 245)
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square_img.paste(rotated, (x_offset, y_offset), rotated) # Paste rotated image on the canvas

    # Convert to RGB (remove alpha) and save
    final_image = square_img.convert("RGB")

    return final_image

    

if __name__ == "__main__":

    # VARIABLES
    rotation_info_path = 'data/rotations_HE/image_rotations_HE.json'
    original_images_path = '../../tissue_alignment/data/images/HE_crops_masked'
    rotated_images_path = 'data/HE_images_rotated'

    # Load the rotation info
    with open(rotation_info_path, 'r') as f:
        rotation_info = json.load(f)

    os.makedirs(rotated_images_path, exist_ok=True) # Create the output directory if it doesn't exist

    for filename, angle in rotation_info.items(): # Iterate through the dictionary with rotation info
        # Construct the full path to the input and output images
        input_path = os.path.join(original_images_path, filename)
        output_path = os.path.join(rotated_images_path, filename)

        # Skip images if marked as "skipped"
        if isinstance(angle, dict) and "skipped" in angle:
            print(f"Skipping {filename} due to 'skipped' flag.") 
            continue
        
        # Rotate the image and save it to the output path
        try:
            img = Image.open(input_path).convert("RGBA")
            rotated_image = rotate_image(img, float(angle))
            rotated_image.save(output_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
