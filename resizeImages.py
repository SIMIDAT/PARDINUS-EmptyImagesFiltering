import os

from PIL import Image

"""
Resizes images in the input_folder to the specified height and width,
and saves them in the output_folder.

Args:
    input_folder (str): Path to the folder containing the images to resize.
    output_folder (str): Path to the folder to save resized images.
    height (int): Desired height of the resized images.
    width (int): Desired width of the resized images.
"""



def resize_images(input_folder, output_folder, height=256, width=384):


    os.makedir(output_folder, exist_ok=True)
    
    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Check if the file is an image
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize((width, height))
                    
                    # Save the resized image in the output folder
                    output_path = os.path.join(output_folder, file_name)
                    resized_img.save(output_path)
                    
                    print(f"Resized and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")



if __name__ == "__main__":

    # Adjust with the name of your data folder
    sourceData = "./Data"
    resizedData = "./ResizedData"
    
    resize_images(sourceData, resizedData)