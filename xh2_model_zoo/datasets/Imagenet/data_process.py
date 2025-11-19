import os
import shutil
import tarfile

import scipy


def un_tar(file_name, output_root="train"):
    # untar zip file to folder whose name is same as tar file
    tar = tarfile.open(file_name)
    names = tar.getnames()

    file_name = os.path.basename(file_name)
    extract_dir = os.path.join(output_root, file_name.split(".")[0])

    # create folder if nessessary
    if os.path.isdir(extract_dir):
        pass
    else:
        os.mkdir(extract_dir)

    for name in names:
        tar.extract(name, extract_dir)
    tar.close()


def untar_traintar(traintar="./traintar"):
    """
    untar images from traintar and save in corresponding folders
    organize like:
    /train
       /n01440764
           images
       /n01443537
           images
        .....
    """
    root, _, files = next(os.walk(traintar))
    for file in files:
        un_tar(os.path.join(root, file))


def move_valimg(val_dir="./val", devkit_dir="./ILSVRC2012_devkit_t12"):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """

    # load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, "data", "meta.mat"))

    ground_truth = open(os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt"))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split(".")[0].split("_")[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset["synsets"][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))
