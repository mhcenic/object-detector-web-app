# DetApp: Object Detector - Web App

[![Build Status](https://travis-ci.org/mhcenic/object-detector-web-app.svg?branch=master)](https://travis-ci.org/mhcenic/object-detector-web-app)
[![Coverage Status](https://coveralls.io/repos/github/mhcenic/object-detector-web-app/badge.svg?branch=master)](https://coveralls.io/github/mhcenic/object-detector-web-app?branch=master)

Web app for image object detection using YOLO algorithm.

You can check the demo [here](https://detv1.herokuapp.com/) (slower version due to server limitations).

### Installation
```bash
git clone https://github.com/mhcenic/object-detector-web-app.git
cd object-detector-web-app
mkdir images

# Add test images to folder.

# [Option 1] Create virtual environment. 
# [Option 2] Install everything globaly.
```

## Quick Start

- Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2 model to a Keras model.
- Run app

```bash
wget http://pjreddie.com/media/files/yolov2.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
./yad2k.py yolov2.cfg yolov2.weights model_data/yolo.h5
./web_app.py
```
See `./yad2k.py --help` for more options.
### Documenation
To generate documentation:
```bash
cd docs
make html
```
