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
#from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.service import Service
#from selenium.webdriver.chrome.webdriver import Options
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils


# Set up Chrome WebDriver
#CHROMEDRIVER_PATH = "/opt/homebrew/bin/chromedriver"  # Change path if needed
#options = Options()
#options.add_argument("--start-maximized")
#service = Service(CHROMEDRIVER_PATH)
#driver = webdriver.Chrome(service=service, options=options)

# File Paths (Change 'yourusername' to match your system)
DOWNLOADS_PATH = f"/Users/lara/Downloads/"
EUROPEANA_IMAGE = os.path.join(DOWNLOADS_PATH, "europeana_image.jpg")
CROPPED_IMAGE = os.path.join(DOWNLOADS_PATH, "cropped_image.jpg")
MONSTER_MASH_MODEL = os.path.join(DOWNLOADS_PATH, "monstermash_3d_model.obj")

API_KEY = "nanodydecti"
QUERY = "sculpture"
SEGMENT_ANYTHING_URL = "https://segment-anything.com/"
MONSTER_MASH_URL = "https://monstermash.zone/"

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
    for i in range(0,len(masks)):
    # https://github.com/facebookresearch/segment-anything/issues/121
    
    #height, width, channels = image.shape
    
    #cutoutImage = image
    
    #for y in range(height):
    #    for x in range(width):
    #        if masks[0]["segmentation"][y][x] == False:
    #            cutoutImage[y,x] = (0, 0, 0)
    
        mask = masks[i]["segmentation"]
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask = mask_utils.encode(np.asfortranarray(mask))
        else:
            print("invalid segmentation format: ", mask)
    
        mask = mask_utils.decode(mask)
        contours, hierarch = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    print(contours)
    #xywh = cv2.boundingRect(contour_list[0])
    #print(xywh)
    #y1_y2_x1_x2 =(y, y+h, x, x+w)
    
        cropped_image = cv2.drawContours(image, contours, -1, (0,0,0), 3)
        
    #    cropped_image = cv2.bitwise_and(image, image, )
        
        
        filename = os.path.join(output_directory, str(i) + '.png')
        cv2.imwrite(filename, cropped_image)
    #driver.get(SEGMENT_ANYTHING_URL)
    #upload_button = WebDriverWait(driver, 10).until(
    #    EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
    #)
    #upload_button.send_keys(EUROPEANA_IMAGE)
    #print("Image uploaded to Segment Anything!")

    #time.sleep(10)

    #download_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Download')]")
    #download_button.click()
    #print(f"Cropped image saved: {CROPPED_IMAGE}")

    #time.sleep(5)

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
