# Accesorize in the dark
Example command for running a targeted physical attack against DVG model with 400 steps.

main.py
--dataset_path
<path to the dataset>/NIR-VIS-2.0
--targeted
--probe-size
40
--batch-size
40
--gallery-index
1
--attack-type
"eyeglass"
--num-of-steps
400
--step-size
1/255
--model
DVG
--protocols
custom_protocols
--mask-init-color
red
--physical

This attack follows the CASIA-NIR-VIS-2.0 dataset protocol. Visit <link> for mor information.

To successfully reproduce a physical attack, the following steps should be taken:

1. Capture NIR Images of the Attacker: Take around 20 Near-Infrared (NIR) images of the attacker, ensuring slight head rotations during capture. The attacker should wear eyeglasses frames with four bright points. This requirement can be bypassed by setting USE_PERSPECTIVE in consts.py to false.

2. Capture a VIS Image: Take a single Visible Light (VIS) image of the attacker.

3. Prepare Images: Crop to 224x224 and center both the NIR and VIS images of the attacker's face.

4. Add to CASIA NIR-VIS-2.0 Dataset: Integrate these images into the CASIA NIR-VIS-2.0 dataset. Ideally, place them in a distinct folder (e.g., s5), while maintaining the existing folder structure of separate NIR and VIS folders.

5. Assign a Unique Label: The new subject should have a unique identifier label within the dataset. A label in the format of 4000<X> is suitable.

6. Create a Dataset Protocol File: Formulate a new dataset protocol file named nir_probe_<X>.txt, where <X> is a chosen identifier that can be set by the --gallery-index argument. Place this file in a 'protocols' folder within the dataset.

7. The 'protocols' folder can differ from the original dataset's folder and its name can be specified using the --protocols argument.

8. The protocol file nir_probe_<X>.txt should first list paths to the new NIR images of the attacker, following by a path to an image of the target. If multiple additional image paths are available, a random target will be selected.

9. A file named vis_gallery_<X>.txt should also be present in the same protocol folder. It should contain paths to VIS images of the attacker, the target, and other subjects you want to include in the gallery.

10. Execute the Attack: Run the attack with the specified command, ensuring the --protocols and --gallery-index arguments are correctly set. This process will generate the attacking eyeglasses in the plot directory.

11. Physical Testing: Print, cut, and wear the attacking eyeglasses.

12. Take images of the attacker with the attacking eyeglasses and run prediction to evaluate their effectiveness.