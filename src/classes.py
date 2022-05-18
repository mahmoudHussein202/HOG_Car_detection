import funct
import numpy as np
from scipy.ndimage.measurements import label

class DetectionInfo():
    def __init__(self,test_images):
        self.max_size = 10
        self.old_bboxes = funct.queue.Queue(self.max_size) 
        self.heatmap = np.zeros_like(test_images[0][:, :, 0])
        
    def get_heatmap(self):
        #self.heatmap = np.zeros_like(test_images[0][:, :, 0])
        if self.old_bboxes.qsize() == self.max_size:
            for bboxes in list(self.old_bboxes.queue):
                self.heatmap = funct.add_heat(self.heatmap, bboxes)
                #self.heatmap = apply_threshold(self.heatmap, 2)
            self.heatmap = funct.apply_threshold(self.heatmap, 20)
        return self.heatmap

    def get_labels(self):
        return label(self.get_heatmap())
        
    def add_bboxes(self, bboxes):
        if len(bboxes) < 1:
            return
        if self.old_bboxes.qsize() == self.max_size:
            self.old_bboxes.get()
        self.old_bboxes.put(bboxes)