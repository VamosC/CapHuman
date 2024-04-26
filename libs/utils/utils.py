import argparse


def get_args():

    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader num of workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--unet_lr', type=float, default=1e-5, help='Unet learning rate')
    parser.add_argument('--root', type=str, default='/cache/exp', help='local exp dir')
    parser.add_argument('--save_interval', type=int, default=5000, help='how often will save checkpoints')
    parser.add_argument('--num_gpus', type=int, default=8, help='how many gpus will be used')
    parser.add_argument('--resume_path', type=str, default='none', help='resume checkpoint')
    parser.add_argument('--dataset_path', type=str, default='celeba')
    parser.add_argument('--precision', type=int, default=16, help='fp32, fp16')
    parser.add_argument('--dataset_type', type=str, default='image_datasets', help='dataset type')
    parser.add_argument('--is_clip_mask', type=int, default=1)
    parser.add_argument('--use_mask', type=int, default=1)
    parser.add_argument('--id_file', type=str, default='facenet', help='id feature')
    parser.add_argument('--text_file', type=str, default='000000.json', help='prompt')
    parser.add_argument('--config', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--drop_text', type=float, default=0)
    parser.add_argument('--mask_loss_weight', type=float, default=1.0)
    parser.add_argument('--sd_locked', type=int, default=1)
    parser.add_argument('--sd_encoder_locked', type=int, default=1)
    parser.add_argument('--cond', type=str, default='normal,albedo,rendered')
    parser.add_argument('--drop_control_cond_t', type=int, default=0)

    args, _ = parser.parse_known_args()

    return args
