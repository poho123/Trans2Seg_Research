import os
import sys
import torch
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import cv2
from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # output folder
    
    output_dir = os.path.join('/kaggle/working/result/', '{}'.format(args.input_img.split('/')[-2]), '{}'.format(args.input_img.split('/')[-1]))
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img)]
    else:
        img_paths = [args.input_img]
    for img_path in img_paths:
        print(img_path)
        image = Image.open(img_path).convert('RGB')
        image=np.array(image)    
        image=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
        images = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, 'trans10kv2')
        print(cfg.DATASET.NAME)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        print("yo",output_dir,outname)
        mask.save(os.path.join(output_dir, outname))


if __name__ == '__main__':
    demo()
