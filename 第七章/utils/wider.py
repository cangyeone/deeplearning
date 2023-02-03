import scipy.io as sio 
import threading 
import queue 
import os 
import cv2 
import numpy as np 
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr
def crop_image(img, boxes, stdsize=24):
    #boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    #img = cv2.imread(im_path)
    height, width, channel = img.shape
    neg_num = 0
    neg_imgs = []# 负样本
    par_imgs = []# 部分样本
    pos_imgs = []# 正样本
    pos_box = []# 正样本位置信息
    par_box = []# 部分样本位置
    neg_box = []# 负样本
    while neg_num < 6:
        size = np.random.randint(40, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        Iou = IoU(crop_box, boxes)
        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
        if np.max(Iou) < 0.3:
            # IOU小于0.3为负样本
            neg_imgs.append(resized_im)
            neg_box.append([0, 0, 0, 0])
            neg_num += 1
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # 忽略像素数比较少的人脸
        if max(w, h) < 12 or x1 < 0 or y1 < 0:
            continue

        # 产生正样本和部分样本
        for i in range(3):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # 随机给定位移
            delta_x = np.random.randint(int(-w * 0.2), int(w * 0.2))
            delta_y = np.random.randint(int(-h * 0.2), int(h * 0.2))

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx1) / float(size)
            offset_y2 = (y2 - ny1) / float(size)

            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                pos_imgs.append(resized_im) 
                pos_box.append([offset_x1, offset_y1, offset_x2, offset_y2])
            elif IoU(crop_box, box_) >= 0.4:
                par_imgs.append(resized_im)
                par_box.append([offset_x1, offset_y1, offset_x2, offset_y2])
    return pos_imgs, par_imgs, neg_imgs, pos_box, par_box, neg_box 


class ImagetDataset():
    def __init__(self, file_dir="data/wider/", num_thread=8, crop_size=24):
        self.crop_size = crop_size
        feedq = queue.Queue(maxsize=100) 
        dataq = queue.Queue(maxsize=100) 
        self.file_dir = file_dir 
        self.feed = threading.Thread(target=self.feed, args=(feedq,))
        self.feed.start()
        self.data = []
        for itr in range(num_thread):
            t = threading.Thread(target=self.process, args=(feedq, dataq))
            self.data.append(t) 
            t.start()
        self.dataq = dataq 
    def batch_data(self, batch_size=60):
        imgs, labs, boxes = [], [], [] 
        for i in range(batch_size):
            img, lab, box = self.dataq.get() 
            imgs.append(img) 
            labs.append(lab) 
            boxes.append(box) 
        imgs = np.stack(imgs, axis=0) 
        labs = np.array(labs) 
        boxes = np.array(boxes) 
        return imgs, labs, boxes 
    def process(self, feedq, dataq):
        while True:
            path, bbox = feedq.get()
            if os.path.exists(path) == False:continue  
            img = cv2.imread(path) 
            #print(img.shape, path, bbox)
            pos_imgs, par_imgs, neg_imgs, pos_box, par_box, neg_box  = crop_image(img, bbox, self.crop_size)
            pos_lab = np.ones([len(pos_box)]) 
            par_lab = -np.ones([len(par_box)]) 
            neg_lab = np.zeros([len(neg_box)])
            image = pos_imgs + par_imgs + neg_imgs 
            boxes = pos_box + par_box + neg_box 
            label = np.concatenate([pos_lab, par_lab, neg_lab])
            sidx = np.arange(len(boxes)) 
            np.random.shuffle(sidx)
            for idx in sidx:
                dataq.put([image[idx], label[idx], boxes[idx]])
    def feed(self, feedq):
        label = sio.loadmat(os.path.join(self.file_dir, "label", "wider_face_train.mat"))
        file_path = [] 
        file_bbox = [] 
        file_list = label["file_list"]
        bbox_list = label["face_bbx_list"]
        n_type = len(bbox_list) 
        for itr_n in range(n_type):
            flist = file_list[itr_n][0]
            fbbox = bbox_list[itr_n][0]
            n_image = len(flist) 
            for itr_m in range(n_image):
                file_name = flist[itr_m][0][0] 
                sfn = file_name.split("_")[1]
                base = os.path.join(self.file_dir, "images", f"{itr_n}--{sfn}")
                file_path.append(os.path.join(base, f"{file_name}.jpg")) 
                file_bbox.append(fbbox[itr_m][0])
        while True:
            for path, box in zip(file_path, file_bbox):
                box[:, 2] += box[:, 0] 
                box[:, 3] += box[:, 1]
                feedq.put([path, box])
    
def main():
    dataset = ImagetDataset() 
    dataset.batch_data()

if __name__ == "__main__":
    main()
