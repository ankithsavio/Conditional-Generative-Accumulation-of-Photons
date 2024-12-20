'''
Inpainting 
mask credits : Repaint, Palette, Free-Form.
'''
import numpy as np
import torch

class inpainting:
    def __init__(self, imgsize = 256, masksize = 128):
        self.imgsize = imgsize
        self.masksize = masksize
    
    def randombbox(self):
        imgsize = self.imgsize
        masksize = np.random.randint(low= self.masksize - (self.masksize * 0.1), high= self.masksize + (self.masksize * 0.3))
        maxr = imgsize - masksize
        t = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        l = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        h = int(masksize)
        w = int(masksize)
        return (t, l, h, w)
    
    def generate_mask(self):
        bbox = self.randombbox()
        imgsize = self.imgsize
        mask = np.zeros((1, imgsize, imgsize), np.float32)
        delta = 15
        h = np.random.randint(delta)
        w = np.random.randint(delta)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,          
                bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        return torch.from_numpy(mask)
    
    def generate_static_mask(self):
        bbox = self.randombbox()
        imgsize = self.imgsize
        mask = np.zeros((1, imgsize, imgsize), np.float32)
        mask[:, bbox[0]:bbox[0]+bbox[2],          
                bbox[1]:bbox[1]+bbox[3]] = 1.
        return torch.from_numpy(mask)
    