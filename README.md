# AI4Culture

## Installing (MacOS)

### Set environment variables

```
export BLENDER_INSTALL_DIR=/Applications/Blender.app/Contents/Resources
export BLENDER_PYTHON=$BLENDER_INSTALL_DIR/4.3/python
# example full path to python shipped with Blender
# /Applications/Blender.app/Contents/Resources/4.3/python/bin/python3.11
export EXTRA_BLENDER_MODULES=PATH_EXTERNAL_BLENDER_DIR/Blender
# example /Users/$USER/Python/Blender
```

### Install dependencies for segmentation step

Only need to install them once. If you want to install more packages on top of these you have to make sure that PYTHONPATH is set to all the subdirs.
```
# Set the environment so that the site-packages are found
export PYTHONPATH=$BLENDER_PYTHON/lib/python3.10/site-packages

$BLENDER_PYTHON/bin/python3.10 -m pip install --target $EXTRA_BLENDER_MODULES/Segment-Anything git+https://github.com/facebookresearch/segment-anything.git
export PYTHONPATH=$PYTHONPATH:$EXTRA_BLENDER_MODULES/Segment-Anything

$BLENDER_PYTHON/bin/python3.10 -m pip install --target $EXTRA_BLENDER_MODULES/PyTorch torch torchvision
export PYTHONPATH=$PYTHONPATH:$EXTRA_BLENDER_MODULES/PyTorch

$BLENDER_PYTHON/bin/python3.10 -m pip install --target $EXTRA_BLENDER_MODULES/OpenCV opencv-python pycocotools
export PYTHONPATH=$PYTHONPATH:$EXTRA_BLENDER_MODULES/OpenCV

$BLENDER_PYTHON/bin/python3.10 -m pip install --target $EXTRA_BLENDER_MODULES/Polygon shapely
export PYTHONPATH=$PYTHONPATH:$EXTRA_BLENDER_MODULES/Polygon
```

### Install dependencies for 3d model step
#### With Selenium
```
# install Chromedriver with HomeBrew
# install Selenium with pip
```

#### With CUDA
```
```
