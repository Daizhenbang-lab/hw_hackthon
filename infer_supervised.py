from datasets_ import load_all_data,load_eval_data
from models import ResNet18
from models.UNet import UNet
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
import os
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# set gpu device
def main(args):
    torch.cuda.set_device(args.gpu)

    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    # load the test set
    test_set = load_all_data(f'data1/H{dataset_idx}_{data_idx}.npy')

    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False)

    #model = ResNet18().cuda()
    model = UNet().cuda()
    model.load_state_dict(torch.load(f'model0_{data_idx}.pth'))


    with torch.no_grad():
        model.eval()
        ys = []
        for input in test_loader:
            print(input.shape)
            input = input.cuda()
            y = model(input)
            ys.append(y.cpu().numpy())
        ys = np.concatenate(ys, axis=0)

    # replace the prediction with the ground truth
    given_data = np.loadtxt(f'data1/Dataset{dataset_idx}InputPos{data_idx}.txt')
    indices = given_data[:, 0].astype(int) - 1
    given_data = given_data[:, 1:]
    ys[indices] = given_data

    # save the result in 4-digit precision
    os.makedirs('results', exist_ok=True)
    np.savetxt(f'results/Dataset{dataset_idx}Output{data_idx}.txt', ys, fmt='%.4f')

def main_(args):

    torch.cuda.set_device(args.gpu)

    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    eval_set = load_eval_data(
        f"data/H{dataset_idx}_{data_idx}.npy",
        f"data/Dataset{dataset_idx}GroundTruth{data_idx}.txt")

    model = UNet().cuda()
    model.load_state_dict(torch.load(f'model0_{data_idx}.pth'))


    eval_loader = DataLoader(
        eval_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    with torch.no_grad():
        model.eval()
        eval_error = 0
        total_samples = 0
        for input, target in eval_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = model.test_loss(output, target)
            eval_error += loss.item() * len(input)
            total_samples += len(input)

        print(f' Eval error: {eval_error / total_samples}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=0)
    parser.add_argument('--data_idx', type=int, default=1)
    args = parser.parse_args()
    main_(args)

    if os.path.exists(f'data1/Round{args.dataset_idx}InputData{args.data_idx}.txt'):
        os.remove(f'data1/Round{args.dataset_idx}InputData{args.data_idx}.txt')
