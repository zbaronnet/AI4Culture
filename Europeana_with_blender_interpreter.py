import sys
# installed selenium in a seperate path with the interpretor of Blender
# Paths may differ based on the machine that you are working
# $ export PATH_TO_BLENDER=/Applications/Blender.app
# $ export PATH_TO_EXTRA_BLENDER_MODULES=/Users/lara/Python/Blender
# $ export PYTHONPATH=PATH_TO_BLENDER/Contents/Resources/4.3/python/lib/python3.11/site-packages
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/Segment-Anything git+https://github.com/facebookresearch/segment-anything.git
# $ export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/Segment-Anything
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/PyTorch torch torchvision
# $ export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/PyTorch
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/OpenCV opencv-python pycocotools
# $ export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/OpenCV
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/Polygon shapely
# $ export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/Polygon
# $ $PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/selenium selenium
# $ export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/selenium

MODELS = "/Users/lara/Python/Blender/Models/"
PATH_TO_SEGMENT_ANYTHING="/Users/lara/Python/Blender/Segment-Anything"
sys.path.append(PATH_TO_SEGMENT_ANYTHING)
PATH_TO_EXTRA_MODULES="/Users/lara/Python/Blender/PyTorch"
sys.path.append(PATH_TO_EXTRA_MODULES)
PATH_TO_OPENCV="/Users/lara/Python/Blender/OpenCV"
sys.path.append(PATH_TO_OPENCV)
PATH_TO_POLYGON="/Users/lara/Python/Blender/Polygon"
sys.path.append(PATH_TO_POLYGON)
PATH_TO_SELENIUM="/Users/lara/Python/Blender/selenium"
sys.path.append(PATH_TO_SELENIUM)

import os
import time
import requests
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon
from shapely.geometry import Point
from PIL import Image

# Set up Chrome WebDriver
CHROMEDRIVER_PATH = "/opt/homebrew/bin/chromedriver"  # Change path if needed
options = Options()
options.add_argument("--start-maximized")
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# File Paths (Change 'yourusername' to match your system)
DOWNLOADS_PATH = f"/Users/lara/Downloads/"
EUROPEANA_IMAGE = os.path.join(DOWNLOADS_PATH, "europeana_image.jpg")
CROPPED_IMAGE = os.path.join(DOWNLOADS_PATH, "Europeana_cut-out_image.png")
HUGGINFACE_MODEL = os.path.join(DOWNLOADS_PATH, "sample.glb")

API_KEY = "my_europeana_key"
QUERY = "sculpture"
HUGGINFACE_URL = "https://huggingface.co/spaces/kushbhargav/3dImages"

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
    im = Image.open(EUROPEANA_IMAGE).convert('RGBA')
    
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

### Step 3: Upload Cropped Image to Huggingface
def upload_to_huggingface():
    driver.get(HUGGINFACE_URL)

    time.sleep(10)

    driver.switch_to.frame(0)
    
    upload_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
    )
    print(CROPPED_IMAGE)
    upload_button.send_keys(CROPPED_IMAGE)
    print("Cropped image uploaded to HuggingFace!")

    time.sleep(5)
    
    inflate_button = driver.find_element(By.XPATH, "//button[text()='Generate']")
    inflate_button.click()
    print("Generate button clicked!")

    time.sleep(60)
    
    inflate_button = driver.find_element(By.XPATH, "//button[text()='Extract GLB']")
    inflate_button.click()
    print("Extract button clicked!")
    
    time.sleep(60)

    export_button = driver.find_element(By.XPATH, "//button[text()='Download GLB']")
    export_button.click()

    time.sleep(5)
    
    if os.path.isfile(HUGGINFACE_MODEL):
        print(f"3D Model saved: {HUGGINFACE_MODEL}")
    else:
        print("No Model was downloaded")


### Step 4: Import into Blender
def import_to_blender():
    bpy.ops.import_scene.gltf(filepath=HUGGINFACE_MODEL)
    print("3D Model imported successfully into Blender!")


### Run Automation
print('hello world')
download_europeana_image()
segment_image()
upload_to_huggingface()
import_to_blender()

driver.quit()
print("Chrome Automation Completed!")
