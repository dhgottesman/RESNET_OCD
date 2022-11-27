import torch
import torch.nn as nn
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
from torchvision import datasets
import resnet


def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':

        data = np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] = get_minibatches
        batch['chunksize'] = chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [], []
        for img, tfrom in zip(images, tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        train_dataset = mnist.MNIST(
            "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
            "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            test_ds.append(deepcopy(batch))
    else:
        model = resnet.ResNet(3)
        data_dir = 'data/cifar10'
        batch_size = 1

        means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
        stds = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

        normalize = transforms.Normalize(
            mean=means,
            std=stds,
        )

        train_transform = transforms.Compose([
            # 4 pixels are padded on each side,
            transforms.Pad(4),
            # a 32×32 crop is randomly sampled from the
            # padded image or its horizontal flip.
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            # For testing, we only evaluate the single
            # view of the original 32×32 image.
            transforms.ToTensor(),
            normalize
        ])

        train_loader, test_loader = get_data_loaders(data_dir,
                                                     batch_size,
                                                     train_transform,
                                                     test_transform,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     pin_memory=True)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            batch = {'input': train_x, 'output': train_label}
            test_ds.append(deepcopy(batch))

    return train_ds, test_ds, model


def get_data_loaders(data_dir,
                     batch_size,
                     train_transform,
                     test_transform,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=False):
    """
    Adapted from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    Utility function for loading and returning train and test
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, set pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - train_transform: pytorch transforms for the training set
    - test_transform: pytorch transofrms for the test set
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - test_loader:  test set iterator.
    """

    # Load the datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return (train_loader, test_loader)
