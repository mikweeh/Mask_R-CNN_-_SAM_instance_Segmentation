This repository contains the code for semainstance segmentation of
small fishes ('Chromis chromis' and 'Coris julis')

The segmentation is done using Mask R-CNN model. The main code does
the training, and also a test, and also it generates a pdf report of
the training.

# Installation

The code is designed to be used with a development environment on
a docker container, launched from VSCode.

It assumes that Docker and Docker Compose are installed. Additionally,
VS Code is installed with the "Dev Container" and "Remote Development" extensions.


## The environment outside and inside the container

Open VS Code in the project's root folder. The structure of the repo is like this:
```
repo
  |_ dataset (void folder)
  |_ src
  |   |_ main.py
  |   |_ ...other codes
  |_ Dockerfile
  |_ docker-compose.yml
  |_ requirements.txt
  |_ README.md
  |_ .env
```

The repo folder should contain a `./src` folder with the codes. `main.py` is the
training code. Each training generates a pdf report at results folder.

The main folder **inside the container** has the same structure, but the dataset
folder contain all the data files. The main folder inside the container is at
path `/ws/` (ws stands for workspace)

The .env file should contain the absolute path to the dataset folder in the
host and also the global variable for the container display. Something like:
```
DATASET = /home/azken/datasets/cystoseira/all
CONTAINER_DISPLAY = 172.28.5.105:30
# You can also define UID and GID here if you need it. The default is 1000.
```
Note that CONTAINER_DISPLAY variable is the same used at the template of
Docker-ROS project. Check there its usage.


## Build the image and run the container

From VSCode, open a terminal in the project folder and build the image:
```
$ docker compose up -d
# It'll run a container called segmentation_cnt from an image called segmentation_img
```

At the bottom left of the VS Code window, there's an icon (the Dev Containers icon).
Click it and select "Attach to a running container". Select the container
`segmentation_cnt`.

You'll see a new instance of VSCode open. Your code is there. You can
modify it and it will be modifies out of the container on the fly. You can also
debug.

When you finish programming you can close the container just by doing:
```
$ docker compose down -v
```
The container will be removed but all your changes will be mapped to the
original folder in the host. Your image will be in perfect condition for
deployment.
