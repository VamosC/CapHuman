from libs.decalib.datasets import detectors
import torchvision.transforms.functional as F
import numpy as np


class BaseFaceRecognition():

    def __init__(self, detector='FAN', device='cuda'):
        if detector == 'FAN':
            self.face_detector = detectors.FAN(device=device)
        elif detector == 'MTCNN':
            self.face_detector = detectors.MTCNN(device=device)

    def detect_face(self, image):
        img = np.array(image).copy()
        bbox, bbox_type = self.face_detector.run(img)
        if len(bbox) != 4:
            print('len(bbox) != 4')
            print('Use original image')
            return image
        image = F.crop(image, int(bbox[1]), int(bbox[0]), int(bbox[3]-bbox[1]), int(bbox[2]-bbox[0]))
        print('detect the face successful!')
        return image
