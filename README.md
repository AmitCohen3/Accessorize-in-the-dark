# Description

This repository contains a code implementation of the paper [Accesorize in the Dark](https://mahmoods01.github.io/files/esorics23-nir-attacks.pdf).

## Setup

### Requirements

We ran the code in this project in Python 3.7. A requierments file is provided to install necessary dependencies (e.g., via `pip install -r requirements.txt`).

### Data

To obtain the CASIA NIR-VIS 2.0 dataset, please visit [this link](http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html) and follow the download instructions. Our experiments adhere strictly to the dataset's established protocols. The dataset is organized in a hierarchical structure as follows:

```bash
CASIA NIR-VIS-2.0
├── s1
│   ├── NIR
│   │   ├── 00001
│   │   ├── 00002
│   │   └── ...
│   └── VIS
│       ├── 00001
│       ├── 00002
│       └── ...
├── s2
├── s3
├── s4
└── protocols
    ├── nir_probe_1.txt
    ├── nir_probe_2.txt
    ├── ...
    ├── vis_gallery_1.txt
    ├── vis_gallery_2.txt
    └── ...
```

To add new subjects, images should be placed in a newly created folder named 's5/', maintaining the same structural format as outlined above.

### Models

To run and evaluate the attack from the paper, you need to download the pretrained model weights and place tham them under the `pretrained/` directory. Links for downloading:

* [LightCNN](https://drive.google.com/uc?export=download&id=1SpMSwbrXcZ9h_KHbbOdpsme1_YXH0eiZ);

* [LightCNN-DVG](https://drive.google.com/uc?export=download&id=1OLepRXZZjtlTPVMMrZkJKU-qPpX7N0I3);

* [Adversarially trained LightCNN-DVG](https://drive.google.com/uc?export=download&id=14JuLy8qCR-_af8kAIMssfU1bYz0TMzGz);

* [ResNest](https://drive.google.com/uc?export=download&id=1HyAj2ohNVKg2R2v_X-RqlLKc4oRR987L).

## Instructions for Producing Physical Attacks

To successfully reproduce a physical attack, the following steps should be taken:

1. Capture NIR images of the attacker: Take around 20 Near-Infrared (NIR) images of the attacker, ensuring slight head rotations during capture. The attacker should wear eyeglasses frames with four bright points. (This requirement can be bypassed by setting USE_PERSPECTIVE in consts.py to false.)

2. Capture a VIS image: Take a single VIS image of the attacker. This is needed to add the subject to the dataset's gallery.

3. Preprocess images: Crop to 224x224 and center both the NIR and VIS images of the attacker's face. It can be done using CropFaces.py in the preprocess folder.

4. Update the CASIA NIR-VIS-2.0 dataset: Integrate these images into the CASIA NIR-VIS-2.0 dataset. Ideally, place them in a distinct folder (e.g., `s5`), while maintaining the existing folder structure of separate NIR and VIS folders.

5. Assign a unique label: The new subject should have a unique identifier label within the dataset. A label in the format of `4000X` should be suitable where X is any number.

6. Following the dataset's protocol, create a dataset protocol file: Formulate a new dataset protocol file named nir_probe_<X>.txt, where <X> is a chosen identifier that can be set by the --gallery-index argument. Place this file in the 'protocols' folder within the dataset.

7. The `protocols` folder can differ from the original dataset's folder and its name can be specified using the --protocols argument.

8. The protocol file nir_probe_X.txt (where X is a chosen identifier) should first list paths to the new NIR images of the attacker, following by a path to an image of the target. If multiple additional image paths are available, a random target will be selected.

9. A file named vis_gallery_X.txt (where X is a chosen identifier) should also be present in the same protocol folder. It should contain paths to VIS images of the attacker, the target, and other subjects you want to include in the gallery.

10. Execute the Attack: Run the attack (use the example command below for reference), ensuring the --protocols and --gallery-index arguments are correctly set. This process will generate the adversarial eyeglasses in the `plot/` directory that will be created automatically.

11. Physical Testing: Print, cut, and wear the adversarial eyeglasses (see the paper for more details).

12. Capture images of the attacker wearing the adversarial eyeglasses and run prediction to evaluate their effectiveness. Prediction can be executed by setting the --predict parameter to True.

## Example Commands

Example command for running a targeted physical attack against the LightCNN-DVG model with 400 steps.

```
python main.py \
  --dataset_path <path to the dataset>/NIR-VIS-2.0 \
  --targeted \
  --probe-size 40 \
  --batch-size 40 \
  --gallery-index 1 \
  --attack-type "eyeglass" \
  --num-of-steps 400 \
  --step-size 1/255 \
  --model DVG \
  --protocols custom_protocols \
  --mask-init-color red \
  --physical
```

Example command for running an untargeted digital attack against the LightCNN model with 400 steps.

```
python main.py \
  --dataset_path <path to the dataset>/NIR-VIS-2.0 \
  --untargeted \
  --probe-size 40 \
  --batch-size 40 \
  --gallery-index 1 \
  --attack-type "eyeglass" \
  --num-of-steps 400 \
  --step-size 1/255 \
  --model LightCNN \
  --protocols custom_protocols \
  --mask-init-color red \
  --non-physical
```

## Citation

If you use this code, please cite our paper:
```
@inproceedings{Cohen23NIR,
  title={Accessorize in the Dark: A Security Analysis of Near-Infrared Face Recognition},
  author={Amit Cohen and Mahmood Sharif},
  booktitle={Proceedings of the 28th European Symposium on Research in Computer Security (ESORICS)},
  year={2023}
}
```
