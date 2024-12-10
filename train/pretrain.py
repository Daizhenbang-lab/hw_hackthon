from datasets import load_labelled_data
# from models import ResNet
from models.resnet1d import ResNet
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import os
import argparse


# set gpu device
def main(args):
    torch.cuda.set_device(args.gpu)

    # load the dataset
    dataset_idx = args.dataset_idx

    train_set = []

    for data_idx in range(1, 4):
        # Load the dataset
        tmp_set = load_labelled_data(
            f'data/Ht{dataset_idx}_{data_idx}.npy',
            f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')
        train_set.append(tmp_set)

    train_set = torch.utils.data.ConcatDataset(train_set)

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2)

    # initialize the model
    model = ResNet().cuda()

    if args.foundation_model is not None and os.path.exists(args.foundation_model):
        model.load_state_dict(torch.load(args.foundation_model, weights_only=True))
        print(f'Foundation model loaded from {args.foundation_model}')
    else:
        print(f'Foundation model not found at {args.foundation_model}.\n Training from scratch.')

    num_epochs = args.num_epochs

    optimizer = optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=5e-2)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        # train the model
        model.train()

        error = 0
        for input, target in train_loader:
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            loss = model.train_loss(model(input), target)
            loss.backward()
            error += loss.item()

            optimizer.step()
            scheduler.step()

        error /= len(train_loader)
        print(f'Epoch {epoch}, Train error: {error}', flush=True)

        torch.save(model.state_dict(), f'checkpoints/pretrained{dataset_idx}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--foundation_model', type=str, default="checkpoints/foundation_model.pth")

    args = parser.parse_args()
    main(args)
