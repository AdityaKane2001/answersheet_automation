from conversion import *
import cv2

from utils import *


dataset = 'test'
ret = RetinaConverter('./gt/%s_annotations.csv'%dataset)
gt = ret()

# predret =
#predret = RetinaConverter('result_csv (2).csv')
#predicted = predret()
# pred_craft = CRAFTConverter('./craft/%s'%dataset)
# craft_annots = pred_craft()
yolopredret = YOLOConverter('yolo/05/yolo_%s_labels/labels'%dataset)
yolo_annots = yolopredret()

# pred_mask = MaskTextConverter('./maskts/%s'%dataset)
# mask_annots = pred_mask()
# print(mask_annots)
# print("MASK")
# print(mask_evaluate(gt,mask_annots,0.5))
evaluate(gt, yolo_annots, 0.5, 0.5)

# print("CRAFT")
#print(craft_evaluate(gt, craft_annots,0.5))
#evaluate(gt,predicted,0.5,0)

# TODO Get all MaskTextSpotter, CRAFT annotations
# TODO Note all mAPs for all things
# TODO run script for all of them
#
# pred = RetinaConverter('')