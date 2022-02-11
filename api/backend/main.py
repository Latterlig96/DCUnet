def main():
    from argparse import ArgumentParser
    import logging
    import uvicorn
    import torch
    from core.model import DcUnet
    from app import app

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(module)s -- %(message)s'))
    log.addHandler(h)

    parser = ArgumentParser()

    parser.add_argument('-m',
                        '--model',
                        dest="model",
                        type=str)

    parser.add_argument('--imgsz',
                        dest="imgsz",
                        nargs='+',
                        default=(256, 192),
                        type=int)

    parser.add_argument('--host',
                        type=str,
                        dest="host",
                        default='0.0.0.0')

    parser.add_argument('-p',
                        '--port',
                        type=int,
                        dest="port",
                        default=8000)
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DcUnet(input_channels=3)
    model.load_state_dict(torch.load(args.model))

    if device == 'cuda':
        model = model.to(device).half()
    
    uvicorn.run(app, host=args.host, port=args.port, use_colors=True)

if __name__ == "__main__":
    main()
