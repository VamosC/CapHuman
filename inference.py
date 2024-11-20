from share import *
import global_config

import os
import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import random

from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms as pth_transforms
from src.cldm.model import create_model, load_state_dict
from src.cldm.ddim_hacked import DDIMSampler

from libs.decalib.deca import DECA
from libs.decalib.utils.config import cfg as deca_cfg
from libs.decalib.datasets import datasets as deca_dataset
from libs.face_parsing import FaceParser
from libs.controlnet.annotator.util import resize_image, HWC3
from libs.controlnet.annotator.openpose import OpenposeDetector

import argparse
from facenet_pytorch import MTCNN, InceptionResnetV1

@torch.no_grad()
def get_id_feat(img_path):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((160, 160)),
        pth_transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_cropped = img_cropped.to(resnet.device)
    else:
        print('fail to detect faces')
        img_cropped = transform(img).to(resnet.device)
    cropped_img = Image.fromarray((einops.rearrange(img_cropped, 'c h w -> h w c') * 127.5 + 127.5).squeeze().cpu().numpy().clip(0, 255).astype(np.uint8))
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding, cropped_img

def draw_facepose(all_lmks, H=512, W=512):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    eps = 0.01
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmk = np.array(lmks)
        # for lmk in lmks:
        lmk = (1+lmk)/2
        x, y = lmk
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def create_inter_data(dataset, modes, meanshape_path=""):

    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    with torch.no_grad():
        code2 = deca.encode(img2)
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")
    tform2 = dataset[-1]["tform"].unsqueeze(0)
    tform2 = torch.inverse(tform2).transpose(1, 2).to("cuda")
    code2["tform"] = tform2

    for i in range(len(dataset) - 1):

        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

        with torch.no_grad():
            code1 = deca.encode(img1)

        # To align the face when the pose is changing
        ffhq_center = None

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = torch.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        for mode in modes:

            code = {}
            for k in code1:
                code[k] = code1[k].clone()

            origin_rendered = None

            if "position" in mode:
                code["tform"] = code2["tform"]
                code["cam"] = code2["cam"]
            if "pose" in mode:
                code["pose"][:, :3] = code2["pose"][:, :3]
            if "lighting" in mode:
                code["light"] = code2["light"]
            if "expression" in mode:
                code["exp"] = code2["exp"]
                code["pose"][:, 3:] = code2["pose"][:, 3:]
            if mode == "all":
                code["pose"] = code2["pose"]
                code["light"] = code2["light"]
                code["exp"] = code2["exp"]
                code["tform"] = code2["tform"]
                code["cam"] = code2["cam"]

            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
            )

            opdict_origin, _ = deca.decode(
                code1,
                render_orig=True,
                original_image=original_image,
                tform=code1["tform"],
            )


            origin_rendered = opdict["rendered_images"].detach()
            clip_image = ((original_image.squeeze().detach().cpu().numpy() * 255).astype("uint8").transpose((1, 2, 0)))
            clip_image = Image.fromarray(clip_image)

            batch = {}
            batch["clip_image"] = clip_image
            batch["image"] = clip_image.copy()
            batch["rendered"] = opdict["rendered_images"].detach()
            batch["normal"] = opdict["normal_images"].detach()
            batch["albedo"] = opdict["albedo_images"].detach()
            batch["mode"] = mode
            batch["normal_origin"] = opdict_origin["normal_images"].detach()
            batch["landmarks2d"] = opdict['landmarks2d'].detach()
            yield batch

def process(input_image, pose_image, prompt, a_prompt="", n_prompt="", num_samples=1, image_resolution=512, ddim_steps=20, strength=1.0, scale=7.0, seed=-1, eta=0.0, modes='position,pose', tau=0.0, controlnet_strength=0.0, controlnet_mode='face,body', *args, **kwargs):

    imagepath_list = [input_image, pose_image]
    dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=image_resolution)
    data = create_inter_data(dataset, modes=[modes])
    if a_prompt != "":
        prompt = prompt + ", " + a_prompt
    print(f"positive prompt = {prompt}")
    print(f"negative prompt = {n_prompt}")

    with torch.no_grad():
        data_batch = next(data)
        image = data_batch["image"]
        clip_image = Image.open(input_image)

        rendered = data_batch["rendered"]
        normal = data_batch["normal"]
        albedo = data_batch["albedo"]
        control = torch.cat([normal, albedo, rendered], dim=1)
        H = W = image_resolution
        pose_control = None
        landmarks2d = data_batch["landmarks2d"]
        landmark = draw_facepose(landmarks2d.squeeze().cpu().numpy().tolist(), H, W)
        landmark_map = Image.fromarray(landmark)
        if hasattr(model, 'pose_control_model'):
            pose_control = torch.zeros(num_samples, 3, H, W)
            if 'body' in controlnet_mode:
                estimate_hand = 'hand' in controlnet_mode
                pose_image = np.array(Image.open(pose_image))
                detected_map, _ = apply_openpose(resize_image(pose_image, image_resolution), hand=estimate_hand)
                detected_map = HWC3(detected_map)
                body_control = torch.from_numpy(detected_map.copy()).float() / 255.0
                body_control = torch.stack([body_control for _ in range(num_samples)], dim=0)
                body_control = einops.rearrange(body_control, 'b h w c -> b c h w')
                pose_control += body_control
            if 'face' in controlnet_mode:
                face_control = torch.from_numpy(landmark).float()/255.0
                face_control = torch.stack([face_control for _ in range(num_samples)], dim=0)
                face_control = einops.rearrange(face_control, 'b h w c -> b c h w')
                pose_control += face_control
            pose_control = pose_control.cuda()

        clip_image, vis_parsing = apply_faceparsing.parse(clip_image)

        if seed == -1:
            seed = random.randint(0, 4294967294)
        seed_everything(seed)

        if global_config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        id_feature, cropped_face = get_id_feat(input_image)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])], "c_crossattn_control": [model.control_cond_stage_model.encode([clip_image])], "c_crossattn_id": [id_feature.unsqueeze(0)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], "c_crossattn_control": [model.control_cond_stage_model.encode([clip_image])], "c_crossattn_id": [id_feature.unsqueeze(0)]}

        if pose_control is not None:
            cond.update({"c_pose_concat": [pose_control]})
            un_cond.update({"c_pose_concat": [pose_control]})

        shape = (4, H // 8, W // 8)

        if global_config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.pose_control_scales = ([controlnet_strength] * 13)

        model.control_scales = ([strength] * 13)
        model.drop_control_cond_t = tau
        print(f'drop_control_cond_t: {model.drop_control_cond_t}')
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     log_every_t=1,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if global_config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        return results

def get_args():

    parser = argparse.ArgumentParser('CapHuman inference pipeline')
    parser.add_argument('--ckpt', type=str, default=None, required=True, help='path to checkpoint')
    parser.add_argument('--sd_ckpt', type=str, default=None, required=True, help='path to other sd ckpt')
    parser.add_argument('--model', type=str, default='./models/cldm_v15.yaml', help='models')
    parser.add_argument('--vae_ckpt', type=str, default=None, help='path to vae ckpt')
    parser.add_argument('--control_ckpt', type=str, default=None)
    parser.add_argument('--controlnet_strength', type=float, default=0.0)
    parser.add_argument('--controlnet_mode', type=str, default='face,body')
    parser.add_argument('--input_image', type=str, default=None)
    parser.add_argument('--pose_image', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--a_prompt', type=str, default="")
    parser.add_argument('--n_prompt', type=str, default="")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--modes', type=str, default='position,pose')
    parser.add_argument('--output_image', type=str, default='examples/output_images/out1.png')

    args = parser.parse_args()
    print(f'Args: {args}')

    return args

if __name__ == '__main__':

    args = get_args()

    # Create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, device='cuda')

    # Create an inception resnet (in eval mode) for ID feature:
    resnet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()

    apply_faceparsing = FaceParser(save_pth='ckpts/face-parsing/79999_iter.pth')

    # Build DECA
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)

    model = create_model(args.model).cpu()
    missing_keys, unexpected_keys = model.load_state_dict(load_state_dict(f'{args.ckpt}', location='cuda'), strict=False)

    if args.sd_ckpt is not None:
        state_dict = load_state_dict(args.sd_ckpt, location='cuda')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if args.vae_ckpt is not None:
        missing_keys, unexpected_keys = model.first_stage_model.load_state_dict(load_state_dict(f'{args.vae_ckpt}', location='cuda'), strict=False)

    if args.control_ckpt is not None and args.controlnet_strength != 0:
        if 'body' in args.controlnet_mode:
            apply_openpose = OpenposeDetector()
        pose_control_model = create_model(f'./models/controlnet/control_v11p_sd15_openpose.yaml').cpu()
        state_dict = load_state_dict(f'{args.control_ckpt}', location='cuda')
        my_state_dict = {}
        for k, v in state_dict.items():
            if 'control_model' in k:
                k = k.replace('control_model.', '')
                my_state_dict[k] = v
        pose_control_model.load_state_dict(my_state_dict, strict=True)
        model.pose_control_model = pose_control_model

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    images = process(**vars(args))
    dir_name = os.path.dirname(args.output_image)
    if dir_name is not None and dir_name != '':
        os.makedirs(dir_name, exist_ok=True)
    Image.fromarray(images[0]).save(args.output_image)
