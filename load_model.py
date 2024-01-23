import os
import torch
from models.light_cnn import LightCNN_29Layers_v2
from models.lightcnn_for_dvg import LightCNN_DVG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(args):
    if args.model.lower() == 'lightcnn':
        model = LightCNN_29Layers_v2(num_classes=80013)
        model = torch.nn.DataParallel(model)
        checkpoint_path = os.path.join(args.pretrained_path, "LightCNN_29Layers.pt")
    elif args.model.lower() == 'dvg':
        model = LightCNN_DVG(num_classes=80013)
        model = torch.nn.DataParallel(model)
        checkpoint_path = os.path.join(args.pretrained_path, "LightCNN_DVG.pt")
    elif args.model.lower() == 'roa':
        model = LightCNN_DVG(num_classes=357)
        model = torch.nn.DataParallel(model)
        checkpoint_path = os.path.join(args.pretrained_path, "LightCNN_DVG_ROA.pt")
    elif args.model.lower() == "resnest":
        from models.resnest.resnest import ResNeSt
        model = ResNeSt(50,0.4,512,7,7)
        checkpoint_path = os.path.join(args.pretrained_path, "ResNeSt.pt")
    else:
        print('Error model type\n')

    model = model.to(device)
    model.eval()

    if checkpoint_path and os.path.isfile(os.path.join(os.getcwd(),checkpoint_path)):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        exit()

    return model
