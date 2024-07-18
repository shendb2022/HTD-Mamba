import time

from CL_Train import train, eval, select_best
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config=None):
    modelConfig = {
        "state": "eval",  # train or select_best, eval
        "epoch": 200,
        "band": 189,
        "batch_size": 80,
        "seed": 1,
        "channel": 16,
        "lr": 1e-4,
        "multiplier": 2.,
        "epision": 10,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_12_.pt",
        "patch_size": 13,
        "m": 5,
        "state_size":16,
        "layer": 1,
        "delta":0.1,
        "dataset": "Sandiego2",
        "path": "datasets/Sandiego2.mat"
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "select_best":
        select_best(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('time is %s'%(end-start))
