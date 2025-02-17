#!/bin/bash

# Installing all dependencies ontop of the Blender Python interpreter
# Please set path to the desired location in your filesystem
export PATH_TO_BLENDER=/Applications/Blender.app
export PATH_TO_EXTRA_BLENDER_MODULES=/Users/lara/Python/Blender
export PYTHONPATH=PATH_TO_BLENDER/Contents/Resources/4.3/python/lib/python3.11/site-packages

# Installing segment-anything
$PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/Segment-Anything git+https://github.com/facebookresearch/segment-anything.git
export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/Segment-Anything

# Installing PyTorch
$PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/PyTorch torch torchvision
export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/PyTorch

# Installing OpenCV
$PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/OpenCV opencv-python pycocotools
export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/OpenCV

# Installing shapely
$PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/Polygon shapely
export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/Polygon

# Installing selenium
$PATH_TO_BLENDER/Contents/Resources/4.3/python/bin/python3.11 -m pip install --target $PATH_TO_EXTRA_BLENDER_MODULES/selenium selenium
export PYTHONPATH=$PYTHONPATH:$PATH_TO_EXTRA_BLENDER_MODULES/selenium
