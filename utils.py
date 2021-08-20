import cv2
from conversion import *
from shapely.geometry import Polygon

eps = 10**-5
def are_same(dict1, dict2):
    dict2_keys = dict2.keys()
    for i in dict1.keys():
        if i not in dict2_keys:
            return False
        if not np.array_equal(dict1[i], dict2[i]):
            return False
    return True


def get_overlap_tuples(matrix):
    overlaps = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i > j:
                if matrix[i, j] > 0 and matrix[i, j] < 1:
                    overlaps.append((i, j))

    return overlaps


def hex_iou(box, hex):
    hhex = Polygon([
        (hex[0], hex[1]),
        (hex[2], hex[3]),
        (hex[4], hex[5]),
        (hex[6], hex[7]),
        (hex[8], hex[9]),
        (hex[10], hex[11])
    ])

    bbox = Polygon([(box[0], box[1]), (box[0], box[3]),
                    (box[2], box[3]), (box[2], box[1])])

    return bbox.intersection(hhex).area / bbox.union(hhex).area


def quad_iou(box, quad):
    qquad = Polygon([(quad[0], quad[1]), (quad[2], quad[3]),
                     (quad[4], quad[5]), (quad[6], quad[7])])

    bbox = Polygon([(box[0], box[1]), (box[0], box[3]),
                    (box[2], box[3]), (box[2], box[1])])

    return bbox.intersection(qquad).area / bbox.union(qquad).area


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    ret_iou = interArea / (float(boxAArea + boxBArea - interArea) + eps)
    return ret_iou


def compute_quad_overlap(a, b):
    overlap_matrix = np.zeros((a.shape[0], b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i, j] = quad_iou(a[i], b[j])
    return overlap_matrix


def compute_overlap(a, b):
    overlap_matrix = np.zeros((a.shape[0], b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i, j] = iou(a[i], b[j])

    return overlap_matrix


def compute_hex_overlap(a, b):
    overlap_matrix = np.zeros((a.shape[0], b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i, j] = hex_iou(a[i], b[j])

    return overlap_matrix


def draw_caption(image, box, caption, mode='up'):
    if mode == 'up':
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    if mode == 'down':
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_predictions(image_name, gt_boxes, pred_boxes):
    img = cv2.imread(os.path.join('valid', image_name))
    img[img < 0] = 0
    img[img > 255] = 255

    if len(gt_boxes) != 0:
        for i in range(len(gt_boxes)):
            bbox = gt_boxes[i]
            try:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = 'GT'
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            except:
                pass

    if len(pred_boxes) != 0:
        for k in range(len(pred_boxes)):
            bbox = pred_boxes[k]
            try:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                draw_caption(img, (x1, y1, x2, y2), 'Questions', mode='down')

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            except:
                pass

    cv2.imwrite(os.path.join('predictions', image_name + '.jpg'), img)


def eliminate_overlaping_boxes(overlap_tuples, annots_boxes):
    final_boxes = [list(i) for i in annots_boxes]
    annot_boxes = [list(i) for i in annots_boxes]

    for j in range(len(overlap_tuples)):
        tup = overlap_tuples[j]
        if annot_boxes[tup[0]][4] > annot_boxes[tup[1]][4]:
            if annot_boxes[tup[1]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[1]])
        else:
            if annot_boxes[tup[0]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[0]])

    return final_boxes


def eliminate_low_conf_boxes(boxes, conf_thresh=0.25):
    final_boxes = [list(i) for i in boxes]

    if len(final_boxes) != 0:

        for j in range(len(final_boxes)):

            if final_boxes[j][4] < conf_thresh:
                final_boxes.pop(j)

            return final_boxes
    return boxes


def box_pruning(boxes, conf_thresh=0.5):
    if boxes.shape != (0,):
        overlap = compute_overlap(boxes, boxes)

        pruned_boxes = eliminate_overlaping_boxes(get_overlap_tuples(overlap), boxes)

        pruned_boxes = eliminate_low_conf_boxes(pruned_boxes, conf_thresh=conf_thresh)

        return np.array(pruned_boxes)
    return boxes


def isempty(annot):
    return np.any(np.isnan(annot))


def get_false_positives(overlaps, thresh):
    less_than_thresh = overlaps < thresh
    non_zeros = overlaps > 0
    return len(np.where(np.logical_and(less_than_thresh, non_zeros) == True)[0])


def get_false_negatives(gt, pred):
    fn = 0
    for i in gt.keys():
        gt_annots = gt[i]
        pred_key = [key for key in pred.keys() if i[:20] in key]

        if not isempty(gt[i]):
            if pred_key == []:
                fn += len(gt[i])
                continue

            pred_annots = pred[pred_key[0]]

            all_overlaps = [compute_overlap(np.array([gt_annots[k]]), pred_annots) for k in range(len(gt_annots))]

            for k in all_overlaps:
                are_zeros = k < 0.5
                fn += len(np.where(np.all(are_zeros))[0])

    return fn

def get_quad_false_negatives(gt, pred):
    fn = 0

    for i in gt.keys():
        gt_annots = gt[i]
        pred_key = [key for key in pred.keys() if i[:20] in key]

        if not isempty(gt[i]):
            if pred_key == []:
                fn += len(gt[i])
                continue

            pred_annots = pred[pred_key[0]]

            all_overlaps = [compute_quad_overlap(np.array([gt_annots[k]]), pred_annots) for k in range(len(gt_annots))]

            for k in all_overlaps:
                are_zeros = k < 0.5
                fn += len(np.where(np.all(are_zeros))[0])

    return fn

def get_hex_false_negatives(gt, pred):
    fn = 0
    for i in gt.keys():
        gt_annots = gt[i]
        pred_key = [key for key in pred.keys() if i[:20] in key]

        if not isempty(gt[i]):
            if pred_key == []:
                fn += len(gt[i])
                continue

            pred_annots = pred[pred_key[0]]

            all_overlaps = [compute_hex_overlap(np.array([gt_annots[k]]), pred_annots) for k in range(len(gt_annots))]

            for k in all_overlaps:
                are_zeros = k < 0.5
                fn += len(np.where(np.all(are_zeros))[0])

    return fn


def get_craft_fn(gt, craft_pred):
    fn = 0
    for i in gt.keys():
        gt_annots = gt[i]
        pred_key = [key for key in craft_pred.keys() if i[:20] in key]
        if not isempty(gt[i]):
            if pred_key == []:
                fn += len(gt[i])
                continue

            pred_annots = craft_pred[pred_key[0]]

            all_overlaps = [compute_overlap(np.array([gt_annots[k]]), pred_annots) for k in range(len(gt_annots))]

            for k in all_overlaps:
                are_zeros = k < 0.5
                fn += len(np.where(np.all(are_zeros))[0])
    return fn


def get_num_bbox(annot_dict):
    num = 0
    for i in annot_dict.keys():
        num += len(annot_dict[i])
    return num


def calculate_map(tp_list, fp_list, num_tp, num_fp, num_annots):
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    recall = tp_cumsum/ num_annots
    precision = tp_cumsum/ (num_tp+ num_fp)

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(gt, predicted, iou_threshold=0.5, conf_thresh=0.05):
    print('Running tests...')
    false_negatives = get_false_negatives(gt, predicted)
    false_positives = 0
    true_positives = 0
    false_positves_list = []
    true_positives_list = []


    num_gt_annots = 0
    num_pred_annots = 0
    #print(gt)


    for i in predicted.keys():
        gt_key = [key for key in gt.keys() if i[:25] in key]

        if gt_key == []:

            continue
        gt_annots = gt[gt_key[0]]

        pred_annots = predicted[i]
        pred_annots = box_pruning(pred_annots, conf_thresh=conf_thresh)

        #draw_predictions(i, gt_annots, pred_annots)

        num_pred_annots += pred_annots.shape[0]

        overlaps = compute_overlap(gt_annots, pred_annots)  # (gt_len,pred_len)

        # print(overlaps)

        if isempty(gt_annots):
            false_positives += pred_annots.shape[0]
            false_positves_list.append(pred_annots.shape[0])

        else:

            num_gt_annots += len(gt_annots)

            true_positives += len(np.where(overlaps > iou_threshold)[0])
            true_positives_list.append(len(np.where(overlaps > iou_threshold)[0]))
            false_positives += get_false_positives(overlaps, iou_threshold)
            false_positves_list.append(get_false_positives(overlaps, iou_threshold))

    print("mAP:", calculate_map(true_positives_list, false_positves_list,true_positives,false_positives, num_gt_annots) )

    print()
    print('True positives: ', true_positives)
    print('False positives: ', false_positives)
    print('False negatives: ', false_negatives)
    print('Num gt annots: ', num_gt_annots)
    print('Num pred annots: ', num_pred_annots)
    print('Precision: ', true_positives / (true_positives + false_positives))
    print('Recall: ', true_positives / (true_positives + false_negatives))


def craft_evaluate(gt, predicted, iou_threshold=0.5):
    print('Running tests...')
    false_negatives = get_quad_false_negatives(gt, predicted)
    false_positives = 0
    true_positives = 0

    num_gt_annots = 0
    num_pred_annots = 0


    for i in predicted.keys():
        gt_key = [key for key in gt.keys() if i[4:25] in key]
        if gt_key == []:
            false_positives += len(predicted[i])
            continue
        gt_annots = gt[gt_key[0]]

        pred_annots = predicted[i]
        # pred_annots = box_pruning(pred_annots, conf_thresh=conf_thresh)

        # draw_predictions(i, gt_annots, pred_annots)

        num_pred_annots += pred_annots.shape[0]

        # (gt_len,pred_len)

        if isempty(gt_annots):
            false_positives += pred_annots.shape[0]

        else:

            num_gt_annots += len(gt_annots)
            overlaps = compute_quad_overlap(gt_annots, pred_annots)

            true_positives += len(np.where(overlaps > iou_threshold)[0])
            false_positives += get_false_positives(overlaps, iou_threshold)
    print('True positives: ', true_positives)
    print('False positives: ', false_positives)
    print('False negatives: ', false_negatives)
    print('Num gt annots: ', num_gt_annots)
    print('Num pred annots: ', num_pred_annots)
    print('Precision: ', true_positives / (true_positives + false_positives))
    print('Recall: ', true_positives / (true_positives + false_negatives))


def mask_evaluate(gt, predicted, iou_threshold=0.5):
    print('Running tests...')
    false_negatives = get_quad_false_negatives(gt, predicted)
    false_positives = 0
    true_positives = 0

    num_gt_annots = 0
    num_pred_annots = 0

    for i in predicted.keys():
        gt_key = [key for key in gt.keys() if i.strip('.txt') in key]
        gt_annots = gt[gt_key[0]]
        if gt_key == []:
            false_positives += len(predicted[i])
            continue
        pred_annots = predicted[i]
        # pred_annots = box_pruning(pred_annots, conf_thresh=conf_thresh)

        # draw_predictions(i, gt_annots, pred_annots)

        num_pred_annots += pred_annots.shape[0]

        # (gt_len,pred_len)

        if isempty(gt_annots):
            false_positives += pred_annots.shape[0]

        else:

            num_gt_annots += len(gt_annots)
            overlaps = compute_quad_overlap(gt_annots, pred_annots)
            true_positives += len(np.where(overlaps > iou_threshold)[0])


            print()

            false_positives += get_false_positives(overlaps, iou_threshold)
    print('True positives: ', true_positives)
    print('False positives: ', false_positives)
    print('False negatives: ', false_negatives)
    print('Num gt annots: ', num_gt_annots)
    print('Num pred annots: ', num_pred_annots)
    print('Precision: ', true_positives / (true_positives + false_positives))
    print('Recall: ', true_positives / (true_positives + false_negatives))
