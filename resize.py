import cv2
import os


input_folder = 'images_filtered'
output_folder = 'dataset/small'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    image = cv2.imread(f'{input_folder}/{filename}')

    height, width, _ = image.shape
    print('Original Dimensions : ', image.shape)

    # Define the desired width and height
    new_width = width // 4
    new_height = height // 4

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print('Final Dimensions : ', resized_image.shape)
    # Save the resized image

    cv2.imwrite(f'{output_folder}/{filename}', resized_image)