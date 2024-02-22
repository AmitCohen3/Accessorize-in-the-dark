import utils
import torch
import torchvision.transforms as transforms
from datasets import NirVisDataset, NirVisDatasetPerSubject

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(args, model, gallery_file_path, probe_file_path):
    # Extracting features from gallery images
    (gallery_features, gallery_names, gallery_dict) = utils.extract_gallery_features(args, model, gallery_file_path)

    # Preparing the dataset and DataLoader based on the type of attack (physical or digital)
    if args.physical_attack:
        probe_dataset = NirVisDatasetPerSubject(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = None  # Process all images of the attacker
        _, true_labels, target_labels, _ = probe_dataset[0]
    else:
        probe_dataset = NirVisDataset(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = args.batch_size

    probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)

    for j, (images, true_labels, target_labels, _) in enumerate(probe_loader):
        images = images.to(device)

        with torch.no_grad():
            probe_features = utils.feature_extract(args, images, model)

        predicted_labels = utils.predict(gallery_features, probe_features, gallery_names)
        print(f"The following subjects:{true_labels.tolist()} were predicted as {predicted_labels}")
