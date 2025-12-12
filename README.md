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
```bash
    cd /usr/src/code 
    bash render_movie.sh [the full path to the bk2file]
```

### Running the integration tool

Get a terminal on the container with 
`docker compose run python_integ_tool bash`

Then:

```bash
    cd /usr/src/code
    bash launch_integ.sh
```

Then in a VNC viewer (I used RealVNC) connect to localhost:5900

For a tutorial see: https://www.youtube.com/watch?v=lPYWaUAq_dY starting at 4:20. The container already has the ROM installed in the appropriate position.


# Apptainer

To run this (on UVA's HPC) we have to use Apptainer instead of Docker as it doesn't require root access.

The setup assumes you have this repository cloned as `/scratch/[your computing id]/AI-Kombatants`
We also assume you have set up a seperate directory for the outputs as `/scratch/[your computing id]/kombat_artifacts`

## The apptainer training environment:

Apptainer is linux only, to work on windows I had to set up WSL2 following the instructions here:

https://deepwiki.com/apptainer/apptainer-admindocs/2.2-windows-and-macos-installation

Apptainer builds a standalone .sif file. Ours clocks in at about 10 GB. 

### Building apptainer

To build, first pick a nice little directory somewhere outside this repo to build into.

In the terminal, first `cd` to **this** directory, then

```bash
apptainer build [/your/output/path/dojo.sif] ./dojo/dojo_apptain.def
```
This creates the dojo.sif file in the given location on your machine.

I then upload the .sif file onto UVA's HPC filesystem into a `kombat_artifacts` folder under my user scratch directory. 

### Using the image (on the UVA HPC)

In a terminal for your session, from this directory

```bash
bash ./code/launch_dojo.sh
```
ASSUMING you have the folders set up under **your** `/scratch` directory,
This will launch the container with this repo mounted to '/dojo' and your `kombat_artifacts` repo mounted to `/kombat_artifacts`

Paths in the scripts are based on being run in the apptainer with these directories available. 
