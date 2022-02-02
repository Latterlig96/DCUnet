import torch
import argparse
from model import DcUnet
import json

parser = argparse.ArgumentParser(description="Export model to torchscript format")

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

parser.add_argument('-b', '--batch-size',
                    dest="batch_size",
                    help="Inference mode batch size",
                    type=int)

parser.add_argument('--save-path',
                    dest="save_path",
                    help="Save output path for torchscript model",
                    type=str)


if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DcUnet(input_channels=3)
    model.load_state_dict(torch.load(args.model))
    if device == 'cuda':
        model.to(device).half()
    
    dummy_input = torch.zeros(args.batch_size, 3, *args.imgsz)
    if device == 'cuda':
        dummy_input = dummy_input.to(device).half()

    ts = torch.jit.trace(model, dummy_input, strict=False)

    if args.save_path is not None:
        ts.save(args.save_path)
    else:
        ts.save('DCUnet.torchscript')
