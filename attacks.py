import time
import utils
import torch
import torchvision.transforms as transforms
from eyeglasses_attack import EyeglassesAttack
from datasets import NirVisDataset, NirVisDatasetPerSubject

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_attack(args, model, gallery_file_path, probe_file_path):
    """
    Performs an attack simulation on a facial recognition model.

    Parameters:
    - args: ArgumentParser object containing various configurations.
    - model: The facial recognition model to be attacked.
    - gallery_file_path: Path to the gallery images (reference images).
    - probe_file_path: Path to the probe images (images to be tested).

    Returns:
    - The success rates before and after the attack, and the targeted attack success rate.
    """
    start = time.time()
    # Initializing performance metrics
    total_num_of_examples, running_targeted_att_succ_rate, running_before_att_succ_rate, running_after_att_succ_rate = 0, 0, 0, 0

    # Extracting features from gallery images
    (gallery_features, gallery_names, gallery_dict) = utils.extract_gallery_features(args, model, gallery_file_path)
    probe_img_list = utils.read_list(probe_file_path)

    # Preparing the dataset and DataLoader based on the type of attack (physical or digital)
    if args.physical_attack:
        probe_dataset = NirVisDatasetPerSubject(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = None # Process all images of the attacker
        num_of_subjects_to_probe = 1
        indices = torch.arange(num_of_subjects_to_probe)
        _, true_labels, target_labels, _ = probe_dataset[0]
        # Logging information about the attacker
        log_attacker_info(args, target_labels, true_labels)
    else:
        probe_dataset = NirVisDataset(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = args.batch_size
        num_of_images_to_probe = min(args.probe_size, len(probe_img_list))
        indices = torch.randperm(len(probe_dataset))[:num_of_images_to_probe]
        # Logging dataset usage info
        log_probe_info(args, num_of_images_to_probe, probe_img_list)

    # Creating a subset of the dataset and a DataLoader for batch processing
    subset = torch.utils.data.Subset(probe_dataset, indices)
    probe_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)

    # Processing each batch of images (for physical attacks, only one batch is used)
    for j, (images, true_labels, target_labels, pose_transformation_matrices) in enumerate(probe_loader):
        pose_transformation_matrices = pose_transformation_matrices.to(device)
        images = images.to(device)

        with torch.no_grad():
            probe_features = utils.feature_extract(args, images, model)
        succ_recognitions_before_attack = utils.count_succ_recognitions(gallery_features, probe_features, gallery_names, true_labels)

        # Preparing for the attack
        target_indices_in_gallery = torch.tensor([list(gallery_dict).index(item) for item in target_labels])
        true_label_indices_in_gallery = torch.tensor([list(gallery_dict).index(item) for item in true_labels])

        if args.attack_type == "eyeglass":
            # Performing eyeglass attack
            mask = utils.load_mask(args.attack_type).to(device)
            eyeglass_attack = EyeglassesAttack(model, mask, gallery_features, true_labels, target_labels, gallery_names, pose_transformation_matrices)
            perturbed_images, delta = eyeglass_attack.execute(args, images, true_label_indices_in_gallery, target_indices_in_gallery)

            # Processing and saving the attacking eyeglasses for physical attacks
            if args.physical_attack:
                utils.process_and_save_glasses(delta, mask)

        else:
            print("Error - attack type is wrong")
            exit()

        # Evaluating the model's performance after the attack
        with torch.no_grad():
            perturbed_imgs_features = utils.feature_extract(args, perturbed_images, model)
            succ_recognitions_after_attack = utils.count_succ_recognitions(gallery_features, perturbed_imgs_features, gallery_names, true_labels)
            succ_targeted_attacks = utils.count_succ_recognitions(gallery_features, perturbed_imgs_features, gallery_names, target_labels) if args.is_targeted else 0

        # Updating performance metrics
        num_of_examples = len(true_labels)
        total_num_of_examples += num_of_examples
        running_after_att_succ_rate += succ_recognitions_after_attack
        running_targeted_att_succ_rate += succ_targeted_attacks
        running_before_att_succ_rate += succ_recognitions_before_attack

    running_after_att_succ_rate /= total_num_of_examples
    running_targeted_att_succ_rate /= total_num_of_examples
    running_before_att_succ_rate /= total_num_of_examples
    end = time.time() - start

    print(f"Attack duration was {end} seconds")
    return running_before_att_succ_rate, running_after_att_succ_rate, running_targeted_att_succ_rate


def log_probe_info(args, num_of_images_to_probe, probe_img_list):
    print(f"Using {num_of_images_to_probe} images out of {len(probe_img_list)} from protocol index {args.gallery_index}")


def log_attacker_info(args, target_labels, true_labels):
    print(f"Using {true_labels.shape[0]} images of the attacker from protocol index {args.gallery_index}")
    if args.is_targeted:
        print(f"Attacker's label is {true_labels[0].item()}, target label is {target_labels[0].item()}")
    else:
        print(f"Attacker's label is {true_labels[0].item()}")
