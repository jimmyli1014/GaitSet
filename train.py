import torch
from model.initialization import initialization
from config import conf
import argparse


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')
opt = parser.parse_args()

# check if device is available
if conf["device"] == "cuda" and not torch.cuda.is_available():  # CUDA
    print("* CUDA is not available, use CPU instead *")
    conf["device"] = "cpu"
elif conf["device"] == "dml":   # DirectML
    import torch_directml
    if torch_directml.is_available():
        dml = torch_directml.device()
        conf["device"] = dml
    else:
        print("* DirectML is not available, use CPU instead *")
        conf["device"] = "cpu"

m = initialization(conf, train=opt.cache)[0]

print("Training START")
m.fit()
print("Training COMPLETE")
