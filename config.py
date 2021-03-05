import pathlib
from dataclasses import dataclass
from typing import Tuple
import yaml


@dataclass
class Config(object): 

    train_dir_path: pathlib.Path
    label_dir_path: pathlib.Path 
    batch_size: int 
    shuffle: bool
    input_dim: Tuple[int, int]
    num_channels: int
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    T_max: int
    eta_min: float

    def __new__(cls, *args, **kwargs):
        init_args, additional_args = {}, {}
        for name, value in kwargs.items():
            if name in cls.__annotations__: 
                init_args[name] = value
            else: 
                additional_args[name] = value
        
        new_cls = super().__new__(cls) 
        new_cls.__init__(**init_args)

        for key, value in additional_args.items():
            setattr(new_cls, key, value)

        return new_cls
    
    @classmethod
    def load_config_class(cls, config_path: str):
        if not isinstance(config_path, str):
            raise TypeError(
                  f"You must provide a config file with manually tuned configuration parameters, but got {config_path} instead")
        with open(config_path, 'rb') as cfg: 
            config = yaml.safe_load(cfg)
        cfg_dict = {}
        for key in config.keys():
            cfg_dict.update(**config[key])
        config_class = cls.__new__(cls, **cfg_dict)
        return config_class
