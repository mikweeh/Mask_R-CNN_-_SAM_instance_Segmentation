This repository contains the code for instance segmentation of small fish
(chromis chromis and coris julis).

This code is used to generate accurate masks (or at least much more
accurate than those we originally had) of these fish in order to have a
better dataset. Generation is very slow; this is not suitable for real-time
inference; it's only an aid for the labeling stage, and even so it's
improvable; but at the point where we are, it's not worth investing more time.

# Installation

The repository is in a container and is used like all the others from the
SRV group (see other previous repos for more details, for example
'Docker_ROSpy311')

Once connected and located in the 'repo' folder, just run `docker compose up`
to start the container and
`docker exec -it segmentation_cnt python3 ./src/main.py` to execute the code.

# Preparation

Before knowing what exactly the code does, it's necessary to have the folder
structure well prepared and make sure that in the '.env' file the 'DATASET'
variable points to the absolute path of your machine where you have the
dataset. This path is the one that will be mapped to the './dataset' path
inside the container ('/ws/dataset')

The dataset folder structure at the beginning must be at least this:
```
 dataset/
     |_ original_coco/       # Folder containing the original COCO annotation file and original images
     |_ original_yolo/       # Folder containing the original yolo annotation files and original images
     |      |_ data.yaml     # Info about the classes
     |      |_ train/        # Folder containing all the image files
     |_ train.txt            # Text file listing image filenames for training
     |_ valid.txt            # Text file listing image filenames for validation
     |_ test.txt             # Text file listing image filenames for test
```


Note that 'original_coco' and 'original_yolo' contain the folders exactly as
they come directly from downloading them from Roboflow.

'train.txt', 'valid.txt' and 'test.txt' are files with one filename per line.
This filename can be in quotes or not, and can have the extension or not (it
works in any case). For example, this is valid:
```
image_001,
'image_002',
'image_003.jpg',
```

# Functionality

The program works by executing the 'main.py' script that is in the '/ws'
folder inside the container. This script contains all the global variables
that can be useful for configuring the behavior. This script can be
launched normally or in inference mode.

## Normal mode

To launch the script in normal mode, just execute it and that's it. This
script makes calls to other scripts that are in the '/ws/utils' folder and
automatically passes all the specified parameters.

Specifically, it does the following:
- First it calls the 'filter_coco.py' script. In this script, the
'original_coco' folder is used to extract the images indicated in the
'train.txt', 'valid.txt' and 'test.txt' files. 3 datasets are generated
in 3 different folders (train, valid and test) with labeling in COCO
format and with the indicated images.

- Then the 'train.py' script is called, which performs the training of a
Mask R-CNN model according to the indicated parameters. The training and
validation datasets are used. A pdf report is generated that is saved in
the './results' folder and contains all the training data. The trained
model is saved in './weights'.

- Then the 'coco2yolo.py' script is called. This script uses the trained
model to make inference on the test dataset images, which are the ones
you'll want to upload back to Roboflow with better labels. The inference
comes out as a matrix, but here it's converted to Yolo polygon format.

- Finally, the 'adapt2rbf.py' script is called, which takes the original
labels from the 'original_yolo' folder, modifies the files so that it keeps
all classes except those you have inferred, generates a new folder with all
the inferred images and labels, attaches the corresponding data.yaml, and
leaves everything ready so you just have to upload that folder to Roboflow.
WARNING, these images will NOT appear as duplicates in Roboflow because one
pixel is modified so they are not identical (and the original name is used
without the roboflow addition).

At the end, therefore, the folder structure you will have will be this:
```
 dataset/
     |_ train/               # Folder containing training images
     |_ valid/               # Folder containing validation images
     |_ test/                # Folder containing test images
     |_ original_coco/       # Folder containing the original COCO annotation file and original images
     |_ original_yolo/       # Folder containing the original yolo annotation files and original images
     |      |_ data.yaml     # Info about the classes
     |      |_ train/        # Folder containing all the image files
     |_ inference            # Folder with the results of the inference process
            |_ images        # Folder with test images with inferenced mask overlapped
            |_ labels        # Folder with original label files of the test images
            |_ labels_full   # Folder with the new generated label files for the test images
     |_ upload/              # Folder containing images labels and data to be uploaded to Roboflow
     |_ train.txt            # Text file listing image filenames for training
     |_ valid.txt            # Text file listing image filenames for validation
     |_ test.txt             # Text file listing image filenames for test
```


## Inference mode

You can launch it like this:
`docker exec -it segmentation_cnt python3 ./src/main.py --mode inference`

You want this for when you already have the trained model and what you want
to do is generate inferences on images that still contain masks in poor
condition. This way you can improve them and upload them back to Roboflow
to inspect and touch them up if necessary.

In this case, training is not executed.

# Others

In the './src/utils' folder there are other scripts that I haven't mentioned.
They are scripts that are used as tools to make checks.

Notable is 'pruebas_masks.py' which is used to see the generated masks.
