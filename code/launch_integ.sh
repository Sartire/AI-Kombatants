#!/bin/bash

# launch Xvfb
Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$! 

export DISPLAY=:99

# Wait for Xvfb to start
sleep 2

echo "Xvfb started"
echo $DISPLAY

# Launch window manager
fluxbox &

# Start x11vnc server
x11vnc -display :99 -nopw -forever -shared &
X11VNC_PID=$!

# Wait for x11vnc to start
sleep 2

echo "x11vnc started"

/stable-retro/gym-retro-integration &

echo "gym-retro-integration started"

wait $X11VNC_PID