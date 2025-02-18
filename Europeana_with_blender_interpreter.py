# Make sure all the dependencies are installed
# If you used the install script you can use the following lines to set the environment
# If you get your dependicies in another way please comment out lines 4-14

# add the user to your machine here. 
# You can find your user by running `echo $USER` in the terminal
machine_user = ''

import sys 
PATH_TO_SEGMENT_ANYTHING=f"/Users/{machine_user}/Python/Blender/Segment-Anything"
sys.path.append(PATH_TO_SEGMENT_ANYTHING)
PATH_TO_EXTRA_MODULES=f"/Users/{machine_user}/Python/Blender/PyTorch"
sys.path.append(PATH_TO_EXTRA_MODULES)
PATH_TO_OPENCV=f"/Users/{machine_user}/Python/Blender/OpenCV"
sys.path.append(PATH_TO_OPENCV)
PATH_TO_POLYGON=f"/Users/{machine_user}/Python/Blender/Polygon"
sys.path.append(PATH_TO_POLYGON)
PATH_TO_SELENIUM=f"/Users/{machine_user}/Python/Blender/selenium"
sys.path.append(PATH_TO_SELENIUM)

import os
import time
import requests
import numpy as np

import bpy

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

# Path to AI models
MODELS = "/Users/machine_user/Python/Blender/Models/"

# Path to chrome driver
# Change path if needed. you can check the path by running `which chromedriver` in your terminal.
CHROMEDRIVER_PATH = "/opt/homebrew/bin/chromedriver"  

# Set OutPut path
OUTPUT_PATH = f"/Users/{machine_user}/Downloads/"
DOWNLOAD_PATH = f"/Users/{machine_user}/Downloads/"

# EuroPeana API key
# Add you API Key
EUROPEANA_KEY = 'API_KEY'

class Create3DModel:
    def __init__(self, image_url_Europeana_API):
        # Set up Chrome WebDriver
        options = Options()
        options.add_argument("--start-maximized")
        service = Service(CHROMEDRIVER_PATH)
        self.driver = webdriver.Chrome(service=service, options=options)
        
        EUROPEANA_IMAGE = self.download_europeana_image(image_url_Europeana_API)
        CROPPED_IMAGE = self.segment_image(EUROPEANA_IMAGE)
        self.HUGGINFACE_3D_MODEL = self.upload_to_huggingface(CROPPED_IMAGE)
        
        self.driver.quit()
        
        return None

    ### Step 1: Download Image from Europeana
    def download_europeana_image(self, image_url):
        image_url = response["items"][0].get("edmPreview")
        self.driver.get(image_url[0])
        try:
            img_data = requests.get(image_url[0]).content
            file_name = os.path.join(OUTPUT_PATH, "europeana_image.jpg")
            with open(file_name, "wb") as img_file:
                img_file.write(img_data)
            print(f"Image saved: {file_name}")
            return file_name
        except:
            print("No valid image was found")

    ### Step 2: Crop Image with Segment Anything
    def segment_image(self, image_file):
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        im = Image.open(image_file).convert('RGBA')

        sam = sam_model_registry["vit_h"](checkpoint=os.path.join(MODELS, "sam_vit_h_4b8939.pth"))
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
 
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
        
        
        filename = os.path.join(OUTPUT_PATH, 'Europeana_cut-out_image' + '.png')
        cut_image = Image.fromarray(im_copy)
        cut_image.save(filename)
        print('cut out image file was created: ', filename)

        return filename

    ### Step 3: Upload Cropped Image to Huggingface
    def upload_to_huggingface(self, image_file):
        HUGGINFACE_URL = "https://huggingface.co/spaces/kushbhargav/3dImages"
        self.driver.get(HUGGINFACE_URL)

        # Wait 10 seconds for the page to be loaded
        time.sleep(10)

        self.driver.switch_to.frame(0)
    
        upload_button = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
        )
        upload_button.send_keys(image_file)
        print("Cropped image uploaded to HuggingFace!")

        time.sleep(5)

        inflate_button = self.driver.find_element(By.XPATH, "//button[text()='Generate']")
        inflate_button.click()
        print("Generate button clicked!")

        # Wait 60 seconds for 3d model to be generated
        time.sleep(60)
    
        inflate_button = self.driver.find_element(By.XPATH, "//button[text()='Extract GLB']")
        inflate_button.click()
        print("Extract button clicked!")

        # Wait 60 seconds for 3d model to be extracted 
        time.sleep(60)

        export_button = self.driver.find_element(By.XPATH, "//button[text()='Download GLB']")
        export_button.click()

        time.sleep(5)

        file_name = os.path.join(DOWNLOAD_PATH, 'sample.glb')
    
        if os.path.isfile(file_name):
            print(f"3D Model saved: {file_name}")
            return file_name
        else:
            print("No Model was downloaded")


print("Let's go and create a 3D model!")


# Set the query
QUERY = 'sculpture'

url = f"https://api.europeana.eu/record/v2/search.json?query={QUERY}&rows=10&wskey={EUROPEANA_KEY}"
response = requests.get(url).json()

if "items" in response and response["items"]:
    image_url = response["items"][0].get("edmPreview")
    
    MODEL_3D = Create3DModel(image_url)

    # Import model into blender
    bpy.ops.import_scene.gltf(filepath=MODEL_3D.HUGGINFACE_3D_MODEL)
    print("3D Model imported successfully into Blender!")

else:
    print("No results for query. Please choose another one and check if your API key is still valid.")


print("Completed!")
