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

from pprint import pprint

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
        open(pdf_loc, 'rb').read(), size=(800, None))]
    return images


class InferenceDataset(Dataset):
    def __init__(self, pdf_loc):
        self.images = self.get_images_tensor(pdf_loc)
        self.scales = self.get_scales()

    def get_images_tensor(self, pdf_loc):
        images = [np.array(i) for i in convert_from_bytes(
            open(pdf_loc, 'rb').read(), size=(800, None))]
        self.raw_images = images
        images = self.resize_ims(images)
        return images

    def _get_scales(self):
        scales = []
        for i in range(len(self.images)):
            imshape = self.images[i].shape
            scales.append(imshape[1]/imshape[0])
        self.scales = np.array(scales)
        return np.array(scales)

    def num_classes(self):
        return 1

    def get_ims_dict(self, idx):
        ret_dict = dict()
        ret_dict['img'] = self.images[idx]
        ret_dict['scale'] = self.scales[idx]
        ret_dict['image_path'] = 'kernel_pdf'
        return ret_dict

    def get_raw_ims(self):
        return self.raw_images

    def get_scales(self):
        return self._get_scales()

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


class PostProcessor(object):
    def __init__(self, images, boxes, conf_thresh=0.5):
        self.images = images
        self.boxes = boxes
        self.conf_thresh = conf_thresh
        self.scales = [i.shape[1]/i.shape[0] for i in self.images]

    def nms(self):
        final_pages = []
        for i in range(len(self.boxes)):
            final_boxes = []
            for j in range(len(self.boxes[i][0])):
                if self.boxes[i][0][j][-1] >= self.conf_thresh:
                    final_boxes.append(self.boxes[i][0][j])
            final_pages.append(final_boxes)
        return final_pages

    def cut_and_save(self):
        boxes = self.nms()
        # print(boxes)
        page = 0
        cuts = [np.ones((1, 800, 3))]
        for i in range(len(self.images)):
            image = self.images[i][:, :800, :]
            # print(image.shape)
            if boxes[i] == []:
                print(image.shape)
                cuts[-1] = np.concatenate((cuts[-1], image))
            else:
                # print(np.vstack(boxes[i]))
                boxesT = np.transpose(np.vstack(boxes[i]))

                ys = boxesT[1]
                print(self.scales)
                sorted_ys = np.round(np.sort(ys) / self.scales[i])
                sorted_ys = sorted_ys.astype(int).tolist()
                sorted_ys.insert(0, 0)
                sorted_ys.append(image.shape[0])
                print(sorted_ys)
                for j in range(len(sorted_ys)-1):
                    # print(self.images[i].shape)
                    # print("Cut:", cuts[-1].shape)

                    if j == 0:
                        cuts[-1] = np.concatenate((cuts[-1],
                                                  image[sorted_ys[j]:sorted_ys[j+1]]), axis=0)
                    cuts.append(image[sorted_ys[j]:sorted_ys[j+1]])
        for cut in cuts:
            cv2.imwrite()
        return cuts

    def __call__(self):
        self.pages = self.nms()
        print(self.cut_and_save())


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

ps = PostProcessor(dataset_val.get_raw_ims(), boxes,)
ps()
