import argparse
from fractions import Fraction

def float_as_str(value):
    if "/" in value:
        return float(Fraction(value))
    return float(value)

class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', default='LightCNN', type=str, metavar='Model',
                            help='Model type: LightCNN,DVG,ROA,RESNEST', choices=["DVG", "LightCNN", "ROA", "RESNEST"])
        parser.add_argument('--num_classes', type=int,
                            metavar='N', help='number of classes')

        parser.add_argument('-g', '--gallery-index', type=int,
                            metavar='N', help='the gallery index to use')

        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 16)')

        parser.add_argument('-b', '--batch-size', default=128, type=int,
                            metavar='N', help='mini-batch size (default: 128). Relevant only for non-universal attacks')

        parser.add_argument('-p', '--probe-size', default=128, type=int,
                            metavar='N', help='number of images to probe (default: 128)')

        parser.add_argument('--num-of-steps', default=400, type=int,
                            metavar='N', help='number of steps for attack')

        parser.add_argument('--mask-init-color', default="red", type=str,
                            help='The mask initialization color (default: Red)',
                            choices=["yellow", "green", "blue", "red", "purple", "cyan", "navy", "grey"])

        group1 = parser.add_argument_group().add_mutually_exclusive_group(required=True)

        group1.add_argument('--physical', dest='physical_attack', action='store_true')
        group1.add_argument('--non-physical', dest='physical_attack', action='store_false')

        group2 = parser.add_argument_group().add_mutually_exclusive_group(required=True)

        group2.add_argument('--targeted', dest='is_targeted', action='store_true')
        group2.add_argument('--untargeted', dest='is_targeted', action='store_false')

        parser.add_argument('--step-size', metavar='N', default=1/255, type=float_as_str, help='The step size for the attack')

        parser.add_argument('--attack-type', default='eyeglass', type=str, help='The type of the attack to perform', choices=['eyeglass', 'sticker', 'facemask', 'pgd', 'pgd2', 'fgsm'])

        ## input if necessary
        parser.add_argument('--dataset_path', default='/Users/amitcohen/Downloads/NIR-VIS-2.0', type=str, metavar='PATH',
                            help='root path of face images (default: none).')

        parser.add_argument('--pretrained_path', default='pretrained', type=str, metavar='PATH',
                            help='path to pretrained checkpoints (default: pretrain)')

        parser.add_argument('--protocols_folder_name', default='protocols', type=str, metavar='PATH',
                            help='list of protocols (default: none).')

        self.args = parser.parse_args()

    def get_args(self):
        return self.args
