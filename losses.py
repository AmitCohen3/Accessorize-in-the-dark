import utils
import torch
import consts
from torch import nn

def vector_wise_cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two sets of vectors.

    Args:
        a: Tensor of vectors (DxB).
        b: Tensor of vectors. (DxB)

    Returns:
        Tensor representing the cosine similarity between each pair of vectors from a and b (Bx1).
    """
    a_norm = a / a.norm(dim=1).view(-1, 1)
    b_norm = b / b.norm(dim=1).view(-1, 1)
    scores = torch.bmm(a_norm.unsqueeze(1), b_norm.unsqueeze(2))
    return scores

def cosine_similarity_loss(features, true_label_features, target_features, is_targeted, gallery_features=None, true_label_indices=None):
    """
    Calculates either targeted or non-targeted cosine similarity loss based on the is_targeted flag.

    Args:
        features: The features of the input samples.
        true_label_features: The features corresponding to the true labels.
        target_features: The features corresponding to the target labels.
        is_targeted: Flag indicating if the loss is targeted.
        gallery_features: The features of all possible labels (gallery).
        true_label_indices: Indices of the true labels in the gallery.

    Returns:
        The calculated cosine similarity loss.
    """
    if is_targeted:
        return targeted_cosine_similarity_loss(features, gallery_features, target_features, true_label_features)
    else:
        return untargeted_cosine_similarity_loss(features, gallery_features, true_label_features, true_label_indices)

def non_printability_score(adv_patches, mask):
    """
    Calculates the non-printability score of adversarial patches.

    Args:
        adv_patches: Adversarial patches.
        mask: Mask indicating the area of the patches.

    Returns:
        The non-printability score.
    """
    lower = consts.NPS_LOW/256
    upper = consts.NPS_HIGH/256
    relu = nn.ReLU()
    lower_loss = relu(lower-adv_patches)
    upper_loss = relu(adv_patches - upper)
    return ((lower_loss*mask).sum() + (upper_loss*mask).sum()) / adv_patches.size(0)

def total_variation_loss(imgs, mask):
    """
    Calculates the total variation loss for images.

    Args:
        imgs: Tensor of images.
        mask: Mask indicating the area for variation calculation.

    Returns:
        The total variation loss.
    """
    w_diff = imgs[None, :, :, :, :-1] - imgs[None, :, :, :, 1:]
    mask_w_diff = mask[None, :, :, :, :-1] - mask[None, :, :, :, 1:]
    mask_w_diff = torch.where(mask_w_diff != 0, torch.zeros(1, device=utils.device), torch.ones(1, device=utils.device))
    w_variance = torch.sum(torch.pow(w_diff * mask_w_diff, 2))
    h_diff = imgs[None, :, :, :-1, :] - imgs[None, :, :, 1:, :]
    mask_h_diff = mask[None, :, :, :-1, :] - mask[None, :, :, 1:, :]
    mask_h_diff = torch.where(mask_h_diff != 0, torch.zeros(1, device=utils.device), torch.ones(1, device=utils.device))
    h_variance = torch.sum(torch.pow(h_diff * mask_h_diff, 2))
    loss = h_variance + w_variance
    return loss / imgs.size(0)

# Untargeted cosine similarity loss
def untargeted_cosine_similarity_loss(features, gallery_features, true_label_features, true_labels_indices):
    """
    Calculates the non-targeted cosine similarity loss.

    sql
    Copy code
    Args:
        features: The features of the input samples.
        gallery_features: The features of all possible labels (gallery).
        true_label_features: The features corresponding to the true labels of the samples.
        true_labels_indices: Indices of the true labels in the gallery.
        reduce: If True, returns the mean loss; otherwise, returns the loss for each sample.

    Returns:
        The calculated non-targeted cosine similarity loss.
    """
    features_norm = features / features.norm(dim=1).view(-1, 1)
    gallery_features_norm = gallery_features.t() / gallery_features.norm(dim=1)
    predict_score = torch.mm(features_norm, gallery_features_norm)
    predict_score[list(range(predict_score.shape[0])), true_labels_indices] -= 1000
    max_index = torch.argmax(predict_score, axis=1)
    predicted_features = torch.index_select(gallery_features, 0, max_index)

    true_label_similarity = vector_wise_cosine_similarity(features, true_label_features)
    prediction_similarity = vector_wise_cosine_similarity(features, predicted_features)
    return -5 * true_label_similarity.mean(dim=0) + 5 *  prediction_similarity.mean(dim=0)

# Trying to get away from all subjects that are closer than the target while getting closer to the target
def targeted_cosine_similarity_loss(features, gallery_features, target_features, true_label_features):
    """
    Calculates the targeted cosine similarity loss.

    This loss function aims to increase the similarity of the input features to a specific target,
    while decreasing their similarity to other subjects, particularly those closer than the target.

    Args:
        features: The features of the input samples.
        gallery_features: The features of all possible labels (gallery).
        target_features: The features corresponding to the target labels.
        true_label_features: The features corresponding to the true labels of the samples.

    Returns:
        The calculated targeted cosine similarity loss.
    """

    # Normalizing the feature vectors
    features_norm = features / features.norm(dim=1).view(-1, 1)
    gallery_features_norm = gallery_features.t() / gallery_features.norm(dim=1)

    # Calculating similarity scores
    predict_score = torch.mm(features_norm, gallery_features_norm)
    true_label_similarity = vector_wise_cosine_similarity(features, true_label_features).squeeze()
    target_similarity = vector_wise_cosine_similarity(features, target_features).squeeze()

    # Identifying scores higher than the target
    higher_than_target = nn.ReLU()(predict_score - target_similarity.unsqueeze(1) + 0.2)

    # Combining the scores to form the loss
    loss = 6 * target_similarity.mean() - 6 * true_label_similarity.mean() - 15 * higher_than_target.mean()
    return loss
