import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
  def __init__(self,txt_path,preproc = None):

        self.imgs_path = []
        self.words = []
        self.preproc = preproc

        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        jo = []
        for line in lines:
            line = line.rstrip()
            #print(line[-1])
            if line[-1]=='g':
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    #print(words)
                    labels.clear()
                    #print(labels)
                path = line
                path = txt_path.replace('wider_face_train_bbx_gt.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
                jo.append(label)
                #print(jo)
        self.words.append(labels)
    

  def __len__(self):
        return len(self.imgs_path)


  def __getitem__(self,index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index][1:]
        annotations = np.zeros((0, 10))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 10))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # scenarious
            annotation[0, 4] = label[4]    # blur
            annotation[0, 5] = label[5]    # expression
            annotation[0, 6] = label[6]    # illumination
            annotation[0, 7] = label[8]    # occlusion
            annotation[0, 8] = label[9]   # pose

            annotation[0,9] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
