import numpy as np 
def IoU(box, boxes):
    """
    计算box与boxes之间的IoU
    其中box为长度为4的向量:x1,y2,x2,y2
    其中boxes为多个长度为4的向量[N, 4]
    返回值为IoU 
    """
    # box面积，加1是补充边界
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes的面积
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 计算内边界点的位置
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # 计算内部边界的高和宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 计算交界部分面积
    inter = w * h
    # 除以总面积
    IoU = inter / (box_area + area - inter)
    return IoU 