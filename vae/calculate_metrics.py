import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import pathlib
import numpy as np
from scipy import linalg
import torchvision
from torch import nn
from torch.nn import functional as F
from model import Classifier

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path of MNIST dataset")
    parser.add_argument(
        "--sample_path", type=str, help="Path to folder containing samples"
    )
    parser.add_argument(
        "--classifier_path", type=str, default="classifier_ckpts/model.pt", help="Path to MNIST classifer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size'
    )
    parser.add_argument(
        "--label_of_dropped_class", type=int, default=0,
        help="Class label of forgotten class (for calculating average prob)"
    )

    args = parser.parse_args()
    return args


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, n=None):
        self.transforms = transforms

        path = pathlib.Path(img_folder)
        self.files = sorted([file for ext in IMAGE_EXTENSIONS
                             for file in path.glob('*.{}'.format(ext))])

        assert n is None or n <= len(self.files)
        self.n = len(self.files) if n is None else n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('L')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def GetImageFolderLoader(path, batch_size):
    dataset = ImagePathDataset(
        path,
        transforms=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    return loader


def calculate_psnr(img1, img2):
    mse = (img1 - img2) ** 2
    mse = torch.mean(mse.view(mse.size(0), -1), dim=1)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.sum()


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(-1, 320)
        return self.fc1(x)


def calculate_fid(act1, act2):
    # Assuming act1 and act2 are numpy arrays of shape (1000, 50)

    # Compute the mean and covariance of the activation vectors
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Compute the squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Compute the product of the covariances and take the square root
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Check if the result is a complex number due to numerical imprecision
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    loader = GetImageFolderLoader(args.sample_path, args.batch_size)
    n_samples = len(loader.dataset)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])),
        batch_size=args.batch_size, shuffle=False)

    selected = None
    while selected is None:
        # select a image with label args.label_of_dropped_class from train_loader
        x, y = next(iter(train_loader))
        for item_x, item_y in zip(x, y):
            if item_y == args.label_of_dropped_class:
                selected = item_x
                break

    model = Classifier().to(device)
    model.eval()
    ckpt = torch.load(args.classifier_path, map_location=device)
    model.load_state_dict(ckpt)
    feature_extractor = FeatureExtractor()
    feature_extractor.conv1 = model.conv1
    feature_extractor.bn1 = model.bn1
    feature_extractor.conv2 = model.conv2
    feature_extractor.bn2 = model.bn2
    feature_extractor.conv2_drop = model.conv2_drop
    feature_extractor.fc1 = model.fc1
    feature_extractor.eval()

    accumulated_psnr = 0
    for data in iter(loader):
        broad_casted_x = selected.unsqueeze(0).repeat(data.size(0), 1, 1, 1).to(device)
        data = data.to(device)
        accumulated_psnr += float(calculate_psnr(data, broad_casted_x))

    accumulated_psnr /= n_samples

    features = None
    with torch.no_grad():
        for data in iter(loader):
            data = data.to(device)
            feature = feature_extractor(data)
            if features is not None:
                features = torch.vstack((features, feature.cpu()))
            else:
                features = feature.cpu()
        selected_feature = feature_extractor(selected.unsqueeze(0).to(device)).cpu()
    fid = calculate_fid(features.numpy(), selected_feature.repeat(features.size(0), 1).numpy())
    print(f'Calculated PSNR: {float(accumulated_psnr)}, Calculated FID: {float(fid)}')
