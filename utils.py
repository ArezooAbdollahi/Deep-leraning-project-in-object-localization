import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import Counter, defaultdict
import torchvision.transforms as standard_transforms
from PIL import Image

def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp

def xywh_to_x1y1x2y2(boxes):
    # input: box in (x,y,w,h) format
    # output: box in (x1,y1,x2,y2) format (representing two corners)
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes

def x1y1x2y2_to_xywh(boxes):
    # input: box in (x1,y1,x2,y2) format 
    # output: box in (x,y,w,h) format
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes

def box_transform(boxes, im_sizes):
    # input: box in (x, y, w, h) format
    # output: normalize (x,y,w,h) in the range of [-1..1]
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = 2 * boxes[:, 0] / im_sizes[:, 0] - 1
    boxes[:, 1] = 2 * boxes[:, 1] / im_sizes[:, 1] - 1
    boxes[:, 2] = 2 * boxes[:, 2] / im_sizes[:, 0] - 1
    boxes[:, 3] = 2 * boxes[:, 3] / im_sizes[:, 1] - 1
    return boxes

def box_transform_inv(boxes, im_sizes):
    # input: box in (x,y,w,h) format normalized in the range of [-1..1]
    # output: convert to (x,y,w,h) in original image coordinates
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = (boxes[:, 0] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 1] = (boxes[:, 1] + 1) / 2 * im_sizes[:, 1]
    boxes[:, 2] = (boxes[:, 2] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 3] = (boxes[:, 3] + 1) / 2 * im_sizes[:, 1]
    return boxes

def compute_IoU(boxes1, boxes2):
    boxes1 = to_2d_tensor(boxes1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = to_2d_tensor(boxes2)
    boxes2 = xywh_to_x1y1x2y2(boxes2)
    
    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy
    
    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert((a1 + a2 - ia <= 0).sum() == 0)
    
    return ia / (a1 + a2 - ia)    

def compute_acc(preds, targets, im_sizes, theta=0.4):
    preds = box_transform_inv(preds.clone(), im_sizes)
    targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    return IoU.sum()/preds.size(0)

class CUBDataset(Dataset):
    # import ipdb; ipdb.set_trace()
    # you need to write some code to handle the custom dataset
    def __init__(self, root = '/home/arezoo/ObjectLocalizationProject/', split='train_images'):
        
        self.root=root
        self.split=split
        self.images_list = []
        self.lables_list = [] 
        self.mean_std = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        self.files = [] 

        self.transform = standard_transforms.Compose([
            standard_transforms.Resize((224,224)),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*self.mean_std)
        ])
 
        file_path = os.path.join(root, self.split + '.txt')

        with open(file_path, 'r') as f: 
            file_list = f.readlines()
            self.files = [x.strip() for x in file_list]

        if self.split == 'train_images':

            file_path_box = os.path.join(root, 'train_boxes' + '.txt')
            with open(file_path_box, 'r') as f: 
                file_list_box = f.readlines()
                self.files_box = [np.array(x.strip().split()).astype(np.float) for x in file_list_box]

    def __getitem__(self, index):
        image = Image.open(os.path.join('images', self.files[index])).convert('RGB')
        image_size= torch.tensor(image.size).float()
        image = self.transform(image)

        if self.split == 'train_images':
            box = self.files_box[index]
            box=box_transform(box, image_size)
            return image, box.view(4), image_size
            
        return image, image_size


    def __len__(self):
        return len(self.files)

    
       