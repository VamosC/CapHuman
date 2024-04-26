import torch
from torchvision import transforms
from .net import sphere
from libs.face_recognition.base_face import BaseFaceRecognition

class CosFace(BaseFaceRecognition):
    def __init__(self, weight):
        super().__init__()

        model = sphere()
        model = model.cuda()
        model.load_state_dict(torch.load(weight))
        model.eval()

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

    def encode(self, img, crop_face=False):
        if crop_face:
            img = self.detect_face(img)

        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        feat = self.model(img)
        return feat
