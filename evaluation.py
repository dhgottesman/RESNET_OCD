import torch
import numpy as np
from diffusion_ocd import Model,Model_Scale
from utils_OCD import overfitting_batch_wrapper,noising,generalized_steps,ConfigWrapper
import argparse
import json
from data_loader import wrapper_dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    '-pd', '--diffusion_model_path', type=str, default = '',
    help='checkpoint path for diffusion model , default <''>')
parser.add_argument(
    '-ps', '--scale_model_path', type=str, default = '',
    help='checkpoint path for scale model , default <''>')
parser.add_argument(
    '-l', '--learning_rate', type=float, default=2e-4, help='learning rate, default <2e-4>')
parser.add_argument(
    '-pb', '--backbone_path', type=str, default = './base_models/checkpoint_tinynerf.pt',
    help='checkpoint path for backbone, default </base_models/checkpoint_tinynerf.pt>')
parser.add_argument(
    '-pc', '--config_path', type=str, default = './configs/train_tinynerf.json',
    help='config path, default </configs/train_tinynerf.json>')
parser.add_argument(
    '-pdts', '--data_test_path', type=str, default = '/data',
    help='test data path, default <''/data''>')
parser.add_argument(
    '-dt', '--datatype', type=str, default = 'tinynerf',
    help='datatype - tinynerf or not, default <tinynerf>')

##########################################################################################################
####################################### Configuration  ###################################################
##########################################################################################################

args = parser.parse_args()
print(args)
with open(args.config_path) as f:
    config = ConfigWrapper(**json.load(f))
torch.manual_seed(123456789)


def evaluate(model, data_loader, device):
    """
    Calculate classification error (%) for given model
    and data set.

    Parameters:

    - model: A Trained Pytorch Model
    - data_loader: A Pytorch data loader object
    """

    y_true = np.array([], dtype=np.int)
    y_pred = np.array([], dtype=np.int)

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))

    error = np.sum(y_pred != y_true) / len(y_true)
    return error


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = args.learning_rate  # learning rate for the diffusion model & scale estimation model

    _, test_loader, base_model = wrapper_dataset(config, args, device)

    # path to desired pretrained base model
    module_path = args.backbone_path
    base_model.load_state_dict(torch.load(module_path))
    base_model = base_model.to(device)

    # OCD + model
    diffusion_model = Model(config=config).cuda()
    scale_model = Model_Scale(config=config).cuda()
    diffusion_model.load_my_state_dict(torch.load(args.diffusion_model_path, map_location=device))
    scale_model.load_my_state_dict(torch.load(args.scale_model_path, map_location=device))

    print("Accuracy of diffusion model: {}".format(evaluate(diffusion_model, test_loader, device)))
    print("Accuracy of scale model: {}".format(evaluate(scale_model, test_loader, device)))
    print("Accuracy of base model: {}".format(evaluate(base_model, test_loader, device)))

