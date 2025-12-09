#!/bin/bash

filename="$1"

Xvfb :99 -screen 0 1024x768x24 & XVFB_PID=$!
export DISPLAY=:99
sleep 3

echo $filename

python3.12 -m retro.scripts.playback_movie $filename
