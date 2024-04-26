import torch
from torchvision import transforms
from .backbones import get_model
from libs.face_recognition.base_face import BaseFaceRecognition

class ArcFace(BaseFaceRecognition):
    def __init__(self, weight, name='r100'):
        super().__init__()

        self.model = get_model(name, fp16=False)
        self.model.load_state_dict(torch.load(weight))
        self.model = self.model.cuda()
        self.model.eval()
        transform = [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

    def encode(self, img, crop_face=False):
        if crop_face:
            img = self.detect_face(img)

        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        feat = self.model(img)
        return feat
