import os
import consts
import numpy as np
import cv2
import time
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms as transforms
from arg_parser import Parser
from datasets import NirVisDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a -> NxD, b -> MxD, result -> NxM
def CosineSimilarity(a, b):
    a_norm = a / a.norm(dim=1).view(-1, 1)
    b_norm = b / b.norm(dim=1).view(-1, 1)
    score=torch.mm(a_norm, b_norm.t())
    return score

# Initialize the directory of the code's plots
def init_plots_dir():
    global timestr
    timestr = time.strftime("%Y%m%d-%H%M%S")

    global plots_dir
    plots_dir = os.path.join("plots", timestr)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    return img_list

def nir_vis_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        pp = path.split(os.path.sep)
        temp = pp[-1].split('.')
        if temp[-1] == 'bmp':
            temp[-1] = 'jpg'
        elif temp[-1] == 'jpg':
            temp[-1] = 'bmp'
        temp = '.'.join(temp)
        pp[-1] = temp
        i_p = os.path.sep.join(pp)
        img = cv2.imread(i_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('image not found')
            print(i_p)
            exit()
    
    return img

def mask_color_init(mask, color):
    color = color.lower()
    out = torch.ones_like(mask) * mask
    rgb_conversion = [0.299,0.587, 0.114]
    rgb_values = [128,128,128] #Initialize to gray by default
    value = 0
    if color == 'yellow':
        rgb_values = [255,255,0]
    if color == 'green':
        rgb_values = [0,255,0]
    if color == 'blue':
        rgb_values = [0,0,255]
    if color == 'red':
        rgb_values = [255,0,0]
    if color == 'purple':
        rgb_values = [128,0,128]
    if color == 'cyan':
        rgb_values = [0,255,255]
    if color == 'navy':
        rgb_values = [0,0,128]
        
    for (conv_val, rgb_val) in zip(rgb_conversion, rgb_values):
        value += conv_val * rgb_val
    
    out = out * value/255.
    return out


def load_mask(position):
    """
    Load the mask corresponding to the attack area. 
    Args:
        position (string): one of ['eyeglass', 'face', 'sticker'].
    Returns:
        mask (torch.Tensor): the mask. Size: 3*128*128.
    """
    path = "mask/224{}.png".format(position)
    print("mask is ", path)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) # From numpy to torch.tensor
    mask[mask>0.5]=1.0
    mask[mask<=0.5]=0.0
    return mask

def count_succ_recognitions(gallery_features, probe_features, gallery_names, probe_names):
    score = CosineSimilarity(gallery_features,probe_features)
    maxIndex = torch.argmax(score, axis=0)
    count = 0

    for i in range(len(maxIndex)):
        if np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
            count += 1
    return float(count)

# Prepare the paths of the protocol files
def prepare_data_paths(dataset_path, protocols_path, protocol_index):
    gallery_file = 'vis_gallery_' + str(protocol_index) + '.txt'
    probe_file = 'nir_probe_' + str(protocol_index) + '.txt'
    full_protocol_path = os.path.join(dataset_path, protocols_path)
    gallery_file_path = os.path.join(full_protocol_path, gallery_file)
    probe_file_path = os.path.join(full_protocol_path, probe_file)

    if not os.path.exists(gallery_file_path):
        print("Could not found gallery file at", gallery_file_path)

    if not os.path.exists(probe_file_path):
        print("Could not found probe file at", probe_file_path)

    return gallery_file_path, probe_file_path


def predict(gallery_features, probe_feature):
    score = CosineSimilarity(gallery_features, probe_feature)
    maxIndex = torch.argmax(score, axis=0)
    return maxIndex


def square_detection(img):
    """
    Detects the four white squares on the frame of the eyeglasses.

    Args:
        img: The image of the eyeglasses.

    Returns:
        A list of square centers.

    Raises:
        RuntimeError: If a glasses patch is not recognized in the image.
    """
    # Preset Regions Of Interest to look for the squares
    roi_corners = [[42, 0], [40, 170], [92, 34], [92, 132]]
    roi_size = 50
    square_centers = []

    for x,y in roi_corners:
        cropped_square = img[0, x:x+roi_size, y:y+roi_size]*255
        max_value = cropped_square.max()
        threshold = int(max_value) - 10

        thresh = cv2.threshold(cropped_square.numpy(), threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=4)

        if thresh.max() == 0:
            #print("Could not recognize glasses patch for image")
            raise RuntimeError("could not recognize glasses patch for image")

        white_indices = np.where(thresh == thresh.max())

        vertical_center = int((white_indices[0][0] + white_indices[0][-1])/2)
        horizontal_center = int((white_indices[1][0] + white_indices[1][-1])/2)
        square_centers.append([y + horizontal_center, x + vertical_center])

    return square_centers

# Detect the 4 squares on the physical eyelgasses image, and find the perspective transformation accordingly
def find_perspective_transform_matrix(img):
    # Reference points in the original eyeglasses mask
    points_src = torch.FloatTensor([[
        [27, 56], [197, 56], [62, 101], [164, 102],
    ]])
    try:
        if consts.USE_PERSPECTIVE:
            points_dst_arr = square_detection(img)
            points_dst_arr = [points_dst_arr]
        else:
            points_dst_arr = points_src
    except RuntimeError:
        points_dst_arr = points_src

    points_dst = torch.FloatTensor(points_dst_arr)
    from kornia.geometry.transform.imgwarp import get_perspective_transform
    T = get_perspective_transform(points_src, points_dst)

    return T

def save_configuration(args):
    file = open(os.path.join(plots_dir, "config.txt"), "w")
    for arg, value in sorted(vars(args).items()):
        file.write("{}: {}\n".format(arg, value))
    file.close()

def feature_extract(args, images, model):
    images = images.to(device)
    if args.model == "RESNEST":
        images = F.interpolate(images, size=112, mode='bilinear')
        images = images.repeat(1, 3, 1, 1)
        features = model(images)
    else:
        images = F.interpolate(images, size=128, mode='bilinear')
        _, features = model(images)
    return features

def extract_gallery_features(args, model, gallery_file):
    gallery_loader = torch.utils.data.DataLoader(
        NirVisDataset(
            root=args.dataset_path,
            file_list=gallery_file,
            transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    features_dim = 512 if args.model == "RESNEST" else 256
    gallery_size = len(read_list(gallery_file))
    gallery_features = torch.zeros(gallery_size, features_dim).to(device)
    gallery_names = torch.zeros(gallery_size)
    total_time = 0.0
    gallery_dict = {}

    with torch.no_grad():
        for j, (images, labels, _, _) in enumerate(gallery_loader):
            start = time.time()
            features = feature_extract(args, images, model)
            gallery_features[j*args.batch_size:(j+1)*args.batch_size] = features
            gallery_names[j*args.batch_size:(j+1)*args.batch_size] = labels
            for l in labels:
                if l.item() in gallery_dict:
                    msg = f"Duplicated label: {l}, you probably added a subject with an existing label"
                    print(msg)
                    raise ValueError(msg)
            dct = dict(zip([t.item() for t in labels], features))
            gallery_dict.update(dct)

            end = time.time() - start
            total_time += end

    gallery_features = gallery_features.to(device)
    print("Gallery batch extraction duration was {} seconds".format(total_time))

    return gallery_features, gallery_names, gallery_dict

def process_and_save_glasses(delta, mask):
    # Brighten the glasses by adding a constant that was derived from physical experiments
    glasses = (delta[0][0] + (consts.PHYSICAL_BRIGHTENING_CONSTANT / 255)) * mask
    # Change the background to be white for printing
    glasses = glasses + 1 - mask
    glasses_path = os.path.join(plots_dir, f"attacking_eyeglasses.png")
    save_image(F.interpolate(glasses, scale_factor=10, mode='nearest'), glasses_path)
