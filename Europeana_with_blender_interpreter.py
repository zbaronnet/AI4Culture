import sys
# installed selenium in a seperate path with the interpretor of Blender
# Paths may differ based on the machine that you are working
# $ export PATH_TO_BLENDER=/Applications/Blender.app
# $ export PATH_TO_EXTRA_BLENDER_MODULES=/Users/lara/Python/Blender
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/selenium selenium
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/Segment-Anything git+https://github.com/facebookresearch/segment-anything.git
# $ /Applications/Blender.app/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/OpenCV opencv-python pycocotools

PATH_TO_EXTRA_MODULES="/Users/lara/Python/Blender"
sys.path.append(PATH_TO_EXTRA_MODULES)
PATH_TO_SEGMENT_ANYTHING="/Users/lara/Python/Blender/Segment-Anything"
sys.path.append(PATH_TO_SEGMENT_ANYTHING)
PATH_TO_OPENCV="/Users/lara/Python/Blender/OpenCV"
sys.path.append(PATH_TO_OPENCV)

import os
import time
import requests
import numpy as np

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon
from shapely.geometry import Point
from PIL import Image

# File Paths (Change 'yourusername' to match your system)
DOWNLOADS_PATH = f"/Users/lara/Downloads/"
EUROPEANA_IMAGE = os.path.join(DOWNLOADS_PATH, "europeana_image.jpg")
CROPPED_IMAGE = os.path.join(DOWNLOADS_PATH, "cropped_image.jpg")
MONSTER_MASH_MODEL = os.path.join(DOWNLOADS_PATH, "monstermash_3d_model.obj")

API_KEY = "my_europeana_key"
QUERY = "sculpture"
#MONSTER_MASH_URL = "https://monstermash.zone/"

### Step 1: Download Image from Europeana
def download_europeana_image():
    url = f"https://api.europeana.eu/record/v2/search.json?query={QUERY}&rows=1&wskey={API_KEY}"
    response = requests.get(url).json()

    if "items" in response and response["items"]:
        image_url = response["items"][0].get("edmPreview")
        print(image_url)
        print(type(image_url))
        if image_url:
            img_data = requests.get(image_url[0]).content
            with open(EUROPEANA_IMAGE, "wb") as img_file:
                img_file.write(img_data)
            print(f"Image saved: {EUROPEANA_IMAGE}")
        else:
            print("No valid image found.")
    else:
        print("No images returned from Europeana.")

### Step 2: Crop Image with Segment Anything
def segment_image():
    image = cv2.imread(EUROPEANA_IMAGE, cv2.IMREAD_COLOR)
    
    sam = sam_model_registry["vit_h"](checkpoint="/Users/lara/Downloads/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    
    output_directory = '/Users/lara/Downloads/'
    
    # source https://github.com/ultralytics/ultralytics/issues/7177
    print("Masks were generated. Creating cropped image from first mask")
    mask = masks[0]["segmentation"]
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        mask = mask_utils.encode(np.asfortranarray(mask))
    else:
        print("invalid segmentation format: ", mask)
    
    mask = mask_utils.decode(mask)

    # Get contours of mask
    # https://github.com/facebookresearch/segment-anything/issues/121
    contours, hierarch = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(contour) for contour in contours] # Convert contours to the correct shape
    contours = [np.atleast_2d(contour) for contour in contours]

    # Create polygon form contours
    polygon = Polygon(contours[0])
    print('polygon created')
    
    # Crop image using Pillow
    # Source https://gist.github.com/yosemitebandit/03bb3ae302582d9dc6be
    pixels = np.array(im)
    im_copy = np.array(im)
    
    for index, pixel in np.ndenumerate(pixels):
        # Unpack the index.
        row, col, channel = index
        # We only need to look at spatial pixel data for one of the four channels.
        if channel != 0:
            continue
        point = Point(col, row)
        if not polygon.contains(point):
            im_copy[(row, col, 0)] = 255
            im_copy[(row, col, 1)] = 255
            im_copy[(row, col, 2)] = 255
            im_copy[(row, col, 3)] = 0
        
        
    filename = os.path.join(output_directory, 'Europeana_cut-out_image' + '.png')
    cut_image = Image.fromarray(im_copy)
    cut_image.save(filename)
    print('cut out image file was created: ', filename)

### Step 3: Upload Cropped Image to Monster Mash
def upload_to_monster_mash():
    driver.get(MONSTER_MASH_URL)

    upload_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
    )
    upload_button.send_keys(CROPPED_IMAGE)
    print("Cropped image uploaded to Monster Mash!")

    time.sleep(5)
    
    inflate_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Inflate')]")
    inflate_button.click()
    print("Inflate button clicked!")

    time.sleep(10)

    export_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Download')]")
    export_button.click()
    print(f"3D Model saved: {MONSTER_MASH_MODEL}")

    time.sleep(5)

### Step 4: Import into Blender
def import_to_blender():
    BLENDER_SCRIPT = f"""
import bpy
bpy.ops.import_scene.obj(filepath="{MONSTER_MASH_MODEL}")
bpy.ops.object.shade_smooth()
print("3D Model imported successfully into Blender!")
"""
    blender_script_path = os.path.join(DOWNLOADS_PATH, "import_model.py")
    
    with open(blender_script_path, "w") as script_file:
        script_file.write(BLENDER_SCRIPT)
    
    os.system(f"blender --background --python {blender_script_path}")

### Run Automation
print('hello world')
download_europeana_image()
segment_image()
#upload_to_monster_mash()
#import_to_blender()

#driver.quit()
#print("Chrome Automation Completed!")
