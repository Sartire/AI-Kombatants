# AI-Kombatants

Using Reinforcement Learning to train agents to play Mortal Kombat II. 

Final project for UVA DS 5004: Applied Reinforcement Learning


## Using the integration tool docker image:


The integration tool container is for running the integration tool, converting replays to videos, and light testing on the environment in jupyter notebooks.
It does not have the ray package installed because that swells the image size to 14 GB

Build with 
`docker compose build`

Using `docker compose up` will launch JupyterLab on port 8889, use the link in the terminal after launch to open it (it will have the auth token)




### Rendering videos

The way the container is mounted by compose, if you drop the bk2 files in ./data for this repo, they will be in the container under /usr/src/data

Docker terminal into the container with:
`docker compose run python_integ_tool bash`

Then: 
```
    cd /usr/src/code 
    bash render_movie.sh [the full path to the bk2file]
```

### Running the integration tool

Get a terminal on the container with 
`docker compose run python_integ_tool bash`

Then:
    cd /usr/src/code
    bash launch_integ.sh

Then in a VNC viewer (I used RealVNC) connect to localhost:5900

For a tutorial see: https://www.youtube.com/watch?v=lPYWaUAq_dY starting at 4:20. The container already has the ROM installed in the appropriate position.


## The apptainer training environment:

Apptainer is linux only, to work on windows I had to set up WSL2 following the instructions here:

https://deepwiki.com/apptainer/apptainer-admindocs/2.2-windows-and-macos-installation

Apptainer builds a standalone .sif file. Ours clocks in at about 10 GB. 

### Building apptainer

To build, first pick a nice little directory somewhere outside this repo to build into.

In the terminal, first `cd` to **this** directory, then

`apptainer build [/your/output/path/dojo.sif] ./dojo/dojo_apptain.def`

This creates the dojo.sif file in the given location on your machine.

I then upload the .sif file onto UVA's HPC filesystem into a `kombat_artifacts` folder under my user scratch directory. 

### Using the image (on the UVA HPC)

In a terminal for your session:

```
cd /path/to/kombat_artifacts
module load apptainer
apptainer shell --contain dojo.sif
```
