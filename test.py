import argparse
import os
import cv2
import torch
import numpy as np
from glob import glob
from model import DcUnet


parser = argparse.ArgumentParser('Test your trained model')

parser.add_argument('-s',
                    '--source',
                    dest="source",
                    help="Input source, might be path or image file",
                    type=str)

parser.add_argument('-m',
                    '--model',
                    dest="model",
                    help="Saved model to perform prediction",
                    type=str)

parser.add_argument('--imgsz',
                    dest="imgsz",
                    help="Image size to resize input frame (compatible with input shape to model)",
                    nargs='+',
                    default=(256, 192),
                    type=int)

parser.add_argument('--save',
                    dest="save",
                    help="Save output, will create runs directory if specified",
                    action="store_true")

parser.add_argument('-v', '--visualize',
                    dest="vis",
                    help="Visualize frames while processing",
                    action="store_true")


def _preprocess(img: np.ndarray, device: str = None) -> torch.Tensor:
    img = img.transpose((2, 0, 1)) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    if device is not None and device == 'cuda':
        return img.to(device).half()
    return img.float()

def _postprocess(img: torch.Tensor, device: str = None) -> np.ndarray:
    if device is not None and device == 'cuda':
        img = img.detach().cpu().numpy()
    else:
        img = img.detach().numpy()
    img = img.squeeze((0, 1)).astype(np.float32)
    img = np.where(img > 0.5, 1, 0).astype(np.float32)
    return img

def _save(file: str, img: np.ndarray) -> None:
    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')
    file = os.path.join('./runs/', file.split('/')[-1])
    cv2.imwrite(file, img)

if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.source.endswith('/'):
        source = glob(os.path.join(args.source, '*'))
    else:
        source = args.source
    
    if args.model.endswith('.torchscript'):
        model = torch.jit.load(args.model)
    else:
        model = DcUnet(input_channels=3)
        model.load_state_dict(torch.load(args.model))
    if device == 'cuda':
        model.to(device).half()
    model.eval()
    
    if isinstance(source, list):
        for file in source:
            img = cv2.imread(file)
            img = cv2.resize(img, args.imgsz)
            img = _preprocess(img, device)
            output = model(img)
            img = _postprocess(output, device)
            if args.save:
                _save(file, img)
            if args.vis:
                cv2.imshow("frame", img)
                cv2.waitKey(0)
    else:
        img = cv2.imread(source)
        img = cv2.resize(img, args.imgsz)
        img = _preprocess(img, device)
        output = model(img)
        img = _postprocess(output, device)
        if args.save:
            _save(source, img)
        if args.vis:
            cv2.imshow("frame", img)
            cv2.waitKey(0)
