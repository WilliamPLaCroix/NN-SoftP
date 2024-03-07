# How to use the LSV-cluster

To log in to the cluster you will need your LSV username and the password. Connect to the contact-node with

```
ssh [USER_NAME]@contact.lsv.uni-saarland.de
```

After asking for your password, you will be connected to the entry node of the LSV cluster. Do not install or run any resource intensive jobs
there. Instead, we need to connect to the submit-node

```
ssh submit
```

Don't forget to clone the repo to your directory on the server.

To actually save any data, you will need to make username and data directories in /data/users/ (/users/USERNAME/data/)

### HTCondor

From this node jobs can be submitted to the queue and HTCondor will assign the requested resources once they are available. In order to specify which resources the job requires a .sub needs to be provided. An example file is run_interactive.sub. Before you can run this file you will need to replace all instances of [USER_NAME] with your username. Then you can submit the job via

```
condor_submit run_interactive.sub -interactive
```
This command will drop you in a live shell on the compute note specified in run_interactive.sub, which is cl8lx in the example. The particular compute node is specified on the line

```
requirements            = (machine == "cl8lx.lsv.uni-saarland.de")
```
Without the -interactive flag, you would have to provide a bash script or command to the .sub file with

```
executable              = /nethome/[USER_NAME]/[SCRIPT_NAME]
```
This script will then be executed on the compute node. An example for such a run script is run_script.sh. The outputs can be observed in previously specified files. In the run.sub file output locations are set with

```
output                  = /data/users/[USER_NAME]/logs/output_$(ClusterId).$(Year)_$(Month)_$(Day)_$(SUBMIT_TIME).log
error                   = /data/users/[USER_NAME]/logs/error_$(ClusterId).$(Year)_$(Month)_$(Day)_$(SUBMIT_TIME).log
log                     = /data/users/[USER_NAME]/logs/log_$(ClusterId).$(Year)_$(Month)_$(Day)_$(SUBMIT_TIME).log
```
Remember to replace [USER_NAME].

#### GPU
The example interactive run file does not request a GPU, so it is best used for small tasks, like building a docker image. GPUs available on the cluster are listed below. To run a job with a GPU have a look at the run.sub file. In particular the following two lines specify how many
GPUs are needed and how much total VRAM:
```
request_GPUs            = 1
requirements            = (GPUs_GlobalMemoryMb >= 16000) && (machine == "cl16lx.lsv.uni-saarland.de")
```

You can 

Titan X GPU clusters:
- cl8lx
- cl9lx
- cl10lx
- cl11lx
- cl12lx

V100 GPU clusters:
- cl16lx (that is the node almost no one uses)

A100 GPU clusters:
- cl17lx
- cl18lx


To view current cluster usage, run this command:
```
condor_q -all -run
```

### Docker

Docker is used for environment management on the LSV-cluster. A docker container essentially provides a virtual machine and thus entirely separate environments to run code. When exiting a docker container all changes are lost and running the container again will produce the exact same environment as it did when running it the first time. So, to have access to a custom python installation, we need to build a docker image with our environment specifications and push it to the docker registry on the cluster. Luckily, this only needs to happen once and the container can be reused for every job. The most recent docker image for our project is already specified in the run.sub file.

```
universe                = docker
docker_image            = docker.lsv.uni-saarland.de/jguertler/nina:v1.1
```
You should be able to just use this image.