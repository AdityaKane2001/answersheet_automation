import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

from conversion import *
import cv2

from utils import *

parser = argparse.ArgumentParser(description='Inference for retinanet.')

parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
parser.add_argument('--mode',default='retina',help="yolo/retinanet",type=str)
parser.add_argument('--model_path', help='Path to model', type=str)
parser.add_argument('--images_path',help='Path to images directory',type=str)
parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
parser.add_argument('--PR_save_path',help='Path to store PR curve image',default=None)
parser.add_argument('--df_save_path')
parser.add_argument('--yolo_labels_dir',default='',type=str,help='Path to labels_dir folder for YOLO detections')
parser = parser.parse_args()

dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))

retinanet=torch.load(parser.model_path)

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

csv_eval.evaluate(dataset_val,df_save_path=parser.df_save_path,retinanet = retinanet,iou_threshold=float(parser.iou_threshold),save_path=parser.PR_save_path)

dataset = 'valid'
ret = RetinaConverter('./annotations/gt/%s_annotations.csv'%dataset)
gt = ret()

# predret =
predret = RetinaConverter('%s_predictions.csv'%(dataset))
predicted = predret()
# pred_craft = CRAFTConverter('./craft/%s'%dataset)
# craft_annots = pred_craft()
# yolopredret = YOLOConverter('yolo/05/yolo_%s_labels/labels'%dataset)
# yolo_annots = yolopredret()
evaluate(gt, predicted, 0.5, 0.5)



