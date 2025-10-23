# AI-Kombatants

Using Reinforcement Learning to train agents to play Mortal Kombat II. 

Final project for UVA DS 5004: Applied Reinforcement Learning


### Current Usage:

`docker compose build`

`docker compose up` will launch JupyterLab on port 8889, use the link in the terminal after launch to open it (it will have the auth token)

### Rendering videos??

Docker terminal into the container, 

cd /usr/src/data
Xvfb :99 -screen 0 1024x768x24
export DISPLAY=:99
python3 -m retro.scripts.playback_movie [the bk2 file]


### running the integration tool?

Problem: the stable-retro source repo has to be available?

I have it forked in into ../stable-retro

docker compose run --rm python bash

cd /stable-retro
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY
make -j$(grep -c ^processor /proc/cpuinfo)