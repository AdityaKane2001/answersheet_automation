import argparse
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval
from utils import *
from conversion import *
import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from skimage.transform import rescale, resize

parser = argparse.ArgumentParser(description='Inference for retinanet.')

# parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
# parser.add_argument('--mode', default='retina',
#                     help="yolo/retinanet", type=str)
# parser.add_argument('--model_path', help='Path to model', type=str)
# parser.add_argument('--class_list_path',
#                     help='Path to classlist csv', type=str)
# parser.add_argument(
#     '--iou_threshold', help='IOU threshold used for evaluation', type=str, default='0.5')
# parser.add_argument(
#     '--PR_save_path', help='Path to store PR curve image', default=None)
# parser.add_argument('--df_save_path')
# parser.add_argument('--yolo_labels_dir', default='', type=str,
#                     help='Path to labels_dir folder for YOLO detections')

parser.add_argument('--pdf_location', '-loc',
                    help="Location id PDF file to process")
parser.add_argument('--model_path', help='Path to model', type=str)
parser = parser.parse_args()


def get_images_tensor(pdf_loc):
    images = [np.array(i) for i in convert_from_bytes(
        open(pdf_loc, 'rb').read(), size=800)]
    return images


class InferenceDataset(Dataset):
    def __init__(self, pdf_loc):
        self.images = self.get_images_tensor(pdf_loc)
        self.scales = self.get_scales()

    def get_images_tensor(self, pdf_loc):
        images = [np.array(i) for i in convert_from_bytes(
            open(pdf_loc, 'rb').read(), size=800)]
        images = self.resize_ims(images)
        return images

    def get_scales(self):
        scales = []
        for i in range(len(self.images)):
            imshape = self.images[i].shape
            scales.append(imshape[1]/imshape[0])

        return np.array(scales)

    def num_classes(self):
        return 1

    def get_ims_dict(self, idx):
        ret_dict = dict()
        ret_dict['img'] = self.images[idx]
        ret_dict['scale'] = self.scales[idx]
        ret_dict['image_path'] = 'kernel_pdf'
        return ret_dict

    def resize_ims(self, images):
        images_arr = []
        for i in images:
            images_arr.append(cv2.resize(i, (640, 640)))
        return torch.from_numpy(np.stack(images_arr)/255.)

    def __getitem__(self, idx):
        return self.get_ims_dict(idx)

    def __len__(self):
        return len(self.scales)

# dataset_val = CSVDataset(parser.csv_annotations_path, parser.class_list_path,
#                          transform=transforms.Compose([Normalizer(), Resizer()]))


dataset_val = InferenceDataset(parser.pdf_location)

retinanet = torch.load(parser.model_path)

use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

if torch.cuda.is_available():
    #retinanet.load_state_dict(torch.load(parser.model_path))
    retinanet = torch.nn.DataParallel(retinanet).cuda()
else:
    retinanet.load_state_dict(torch.load(parser.model_path))
    retinanet = torch.nn.DataParallel(retinanet)

retinanet.training = False
retinanet.eval()
retinanet.module.freeze_bn()

boxes = csv_eval.get_detections(dataset_val, retinanet)
print(boxes)
