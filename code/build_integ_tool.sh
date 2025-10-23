cd /stable-retro
cmake . -DBUILD_UI=ON -UPYLIB_DIRECTORY -DPython3_EXECUTABLE=/usr/bin/python3.12 -DPython_INCLUDE_DIR=/usr/include/python3.12
make -j$(grep -c ^processor /proc/cpuinfo)
cd /usr/src/code
