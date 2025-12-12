#!/bin/bash

username=$(whoami)

host_repo_path="/scratch/$username/AI-Kombatants"
host_output_path="/scratch/$username/kombat_artifacts"

# launch the apptainer
module load apptainer
module load pytorch
apptainer shell --nv --contain --bind $host_repo_path:/dojo --bind $host_output_path:/kombat_artifacts $host_output_path/dojo.sif
