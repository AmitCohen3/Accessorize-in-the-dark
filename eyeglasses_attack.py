import kornia
import torch
import consts
import losses
from models import image_transform_layers
from utils import mask_color_init, feature_extract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeglassesAttack:
    """
    Class representing an adversarial attack using eyeglasses on a facial recognition model.

    Attributes:
    - model: The facial recognition model being attacked.
    - mask: The mask representing the eyeglasses.
    - gallery_features: Features of gallery images used by the facial recognition model.
    - true_names: Actual labels of the subjects.
    - target_names: Target labels for the attack.
    - gallery_names: Names associated with the gallery images.
    - pose_transformation_matrices: Matrices for pose transformations of the eyeglasses.
    """

    def __init__(self, model, mask, gallery_features , true_names, target_names, gallery_names, pose_transformation_matrices):
        self.model = model
        self.mask = mask
        self.gallery_features = gallery_features
        self.true_names = true_names
        self.target_names = target_names
        self.gallery_names = gallery_names
        self.pose_transformation_matrices = pose_transformation_matrices

    def execute(self, args, inputs, true_labels_idxs_in_gallery, target_labels_idxs_in_gallery):
        """
        Executes the adversarial attack.

        Parameters:
        - args: Configuration arguments for the attack.
        - inputs: Input images to be attacked.
        - true_labels_idxs_in_gallery: Indices of true labels in the gallery.
        - target_labels_idxs_in_gallery: Indices of target labels in the gallery.

        Returns:
        - Perturbed images and the delta (attacking eyeglasses).
        """
        init_mask = mask_color_init(self.mask, args.mask_init_color)
        mask_perspective = (self._apply_perspective_transformation(self.mask, inputs.size(-1)) / 255. > 0.5).float()

        delta = self._init_delta(args, init_mask, inputs)

        # Iteratively update delta through optimization steps
        for t in range(args.num_of_steps):
            delta_perspective = self._apply_perspective_transformation(delta, inputs.size(-1)) / 255.
            # Loss calculation over random affine and rotation transformations
            final_loss = self._calculate_loss(args, delta, delta_perspective, inputs, mask_perspective,
                                              target_labels_idxs_in_gallery, true_labels_idxs_in_gallery)
            # Backprop for gradient calculation
            final_loss.backward()
            # Update delta based on calculated loss
            self._update_delta(args, delta)

        perturbed = self._generate_perturbed_images(delta_perspective, inputs, mask_perspective)
        return perturbed, delta.detach()

    def _generate_perturbed_images(self, delta_perspective, inputs, mask_perspective):
        perturbed = (inputs * (1 - mask_perspective) + delta_perspective * mask_perspective).clamp(0, consts.EPSILON)
        return perturbed.detach()

    def _update_delta(self, args, delta):
        delta.data = (delta + args.step_size * delta.grad.detach().sign()).clamp(0, consts.EPSILON)
        delta.data = delta.detach() * self.mask  # Verify that only mask pixels are perturbed
        delta.grad.zero_()

    def _calculate_loss(self, args, delta, delta_perspective, inputs, mask_perspective, target_labels_idxs_in_gallery,
                        true_labels_idxs_in_gallery):
        """
        Calculates the loss for the attack iteration.

        Args:
            args: Configuration arguments for the attack.
            delta: Current perturbation tensor.
            delta_perspective: Perspective-transformed delta.
            inputs: Input images.
            mask_perspective: Perspective-transformed mask.
            target_labels_idxs_in_gallery: Indices of target labels in the gallery.
            true_labels_idxs_in_gallery: Indices of true labels in the gallery.

        Returns:
            The calculated loss for the current attack iteration.
        """
        idx = self._select_random_subset(args, inputs)
        robustness_iters = consts.ITERS_CNT_FOR_EOT if args.physical_attack else 1
        final_loss = 0.

        for _ in range(robustness_iters):
            delta_transformed, mask_transformed = self._transform_delta_and_mask(args, delta_perspective,mask_perspective)
            temp_imgs = ((inputs[idx]) * (1 - mask_transformed[idx]) + delta_transformed[idx]).clamp(0., consts.EPSILON)
            loss = self._compute_cosine_similarity_loss(args, idx, target_labels_idxs_in_gallery, temp_imgs, true_labels_idxs_in_gallery)

            # Average the loss over the transformations
            final_loss += loss / robustness_iters

        # Incorporate additional loss terms
        final_loss = self._add_tv_and_nps_loss_terms(delta, final_loss)
        return final_loss

    def _add_tv_and_nps_loss_terms(self, delta, final_loss):
        # Add total variation to the final loss
        tv_loss = losses.total_variation_loss(delta, self.mask)
        final_loss -= 1 * tv_loss * consts.TV_WEIGHT
        # Add non-printability score to the final loss
        nps = losses.non_printability_score(delta, self.mask)
        final_loss -= 1 * nps * consts.NPS_WEIGHT
        return final_loss

    def _compute_cosine_similarity_loss(self, args, idx, target_labels_idxs_in_gallery, temp_imgs, true_labels_idxs_in_gallery):
        features = feature_extract(args, temp_imgs, self.model)
        loss = losses.cosine_similarity_loss(
            features,
            self.gallery_features[true_labels_idxs_in_gallery][idx],
            self.gallery_features[target_labels_idxs_in_gallery][idx],
            args.is_targeted, self.gallery_features,
            true_label_indices=true_labels_idxs_in_gallery[idx])
        return loss

    def _transform_delta_and_mask(self, args, delta_perspective, mask_perspective):
        """
        Applies transformations to the delta and mask tensors.

        This method performs a series of transformations on the delta (perturbation)
        and mask tensors. It applies affine and rotational transformations, which can
        be either random or identity transformations based on the type of attack (physical
        or digital). Additionally, for physical attacks, random noise is added to the
        delta for robustness. The transformed delta is then clamped to legal values and
        ensured to stay within the mask boundaries.

        Args:
            args: Configuration arguments, used to determine the type of attack and transformations.
            delta_perspective: The perspective-transformed delta tensor.
            mask_perspective: The perspective-transformed mask tensor.

        Returns:
            Tuple of transformed delta tensor and transformed mask tensor.
            The delta tensor is transformed by affine and rotation transformations,
            with added noise for physical attacks and clamped to valid values.
            The mask tensor is transformed similarly and binarized.
        """
        # Define random (or identity) affine and rotation transformations
        affine_transform = image_transform_layers.AffineTransform(identity_transform=not args.physical_attack)
        rotation_transform = image_transform_layers.RotationTransform(identity_transform=not args.physical_attack)
        # Transform the mask using the drawn transformations, and binarize it afterwards
        mask_transformed = (rotation_transform(affine_transform(mask_perspective)) > 0.5).float()
        # Transform the delta using the drawn transformations
        delta_transformed = rotation_transform(affine_transform(delta_perspective))
        # Add random noise.
        delta_transformed += args.physical_attack * torch.normal(0, 0.04, size=delta_transformed.shape).to(device)
        # Clamp the delta to have legal values
        delta_transformed = delta_transformed.clamp(0, consts.EPSILON)
        # Make sure delta stays in the mask boundaries
        delta_transformed *= mask_transformed
        return delta_transformed, mask_transformed

    def _select_random_subset(self, args, inputs):
        # Use a random subset of the inputs, improves generalization
        perm = torch.randperm(inputs.size(0))
        idx = perm[:consts.SUBSET_SIZE] if args.physical_attack else perm
        return idx

    def _init_delta(self, args, init_mask, inputs):
        if args.physical_attack:
            # Create one general eyeglasses frame
            delta = torch.ones_like(inputs[:1]).to(device)
        else:
            delta = torch.ones_like(inputs).to(device)
        delta *= init_mask
        delta.requires_grad = True
        return delta

    def _apply_perspective_transformation(self, tensor_to_transform, target_size):
        return kornia.warp_perspective(
            tensor_to_transform.expand(self.pose_transformation_matrices.size(0), -1, -1, -1) * 255.,
            self.pose_transformation_matrices[:, 0, :, :],
            dsize=(target_size, target_size),
            align_corners=False)