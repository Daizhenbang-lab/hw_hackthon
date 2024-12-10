from datasets import load_all_data
from models import ResNet
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
import os


# set gpu device
def main(args):
    torch.cuda.set_device(args.gpu)

    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    # load the test set
    test_set = load_all_data(f'data/H{dataset_idx}_{data_idx}.npy')

    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=4)

    model = ResNet().cuda()
    model.load_state_dict(torch.load(f'model{dataset_idx}_{data_idx}.pth'))

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        model.eval()
        # model.train() # BN adaptation
        ys = []
        for input in test_loader:
            print(input.shape)
            input = input.cuda()
            y = model(input)
            ys.append(y.cpu().numpy())
        ys = np.concatenate(ys, axis=0)

    # replace the prediction with the ground truth
    given_data = np.loadtxt(f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')
    indices = given_data[:, 0].astype(int) - 1
    given_data = given_data[:, 1:]
    ys[indices] = given_data

    # save the result in 4-digit precision
    os.makedirs('results', exist_ok=True)
    np.savetxt(f'results/Dataset{dataset_idx}Output{data_idx}.txt', ys, fmt='%.4f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=1)
    parser.add_argument('--data_idx', type=int, default=1)
    args = parser.parse_args()
    main(args)

