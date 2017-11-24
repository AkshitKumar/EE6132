#!/bin/bash

if [[ $OSTYPE == darwin* ]]; then
    curl -O http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
elif [[ $OSTYPE == linux-gnu ]]; then
    wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
fi

unzip sketches_png.zip && rm sketches_png.zip

wget -O crop_resize128.zip https://www.dropbox.com/s/eclcxeue0oq8lgs/crop_resize128.zip?dl=1

unzip crop_resize128.zip && rm crop_resize128.zip