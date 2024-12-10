from datasets import load_labelled_data, load_eval_data
# from models import ResNet
from models.resnet1d import ResNet
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import os
import argparse


def main(args):
    torch.cuda.set_device(args.gpu)

    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    # train_set = []

    # for i in range(1, 4):
    #     # Load the dataset
    #     tmp_set = load_labelled_data(
    #         f'data/Ht{dataset_idx}_{i}.npy',
    #         f'data/Dataset{dataset_idx}InputPos{i}.txt')
    #     train_set.append(tmp_set)

    # train_set = torch.utils.data.ConcatDataset(train_set)

    train_set = load_labelled_data(
        f'data/Ht{dataset_idx}_{data_idx}.npy',
        f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')

    # load the evaluation dataset
    eval_set = load_eval_data(
        f"data/Ht{dataset_idx}_{data_idx}.npy",
        f"data/Dataset{dataset_idx}GroundTruth{data_idx}.txt")

    num_epochs = args.num_epochs

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2)
    eval_loader = DataLoader(
        eval_set, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=2)

    # prepare the model
    model = ResNet().cuda()

    os.makedirs('checkpoints', exist_ok=True)
    args.checkpoint = f"checkpoints/{args.checkpoint}"

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print('Loaded the model from the checkpoint')
    else:
        print('Training the model from scratch')

    optimizer = optim.AdamW(
        model.parameters(), lr=1e-2, weight_decay=5e-2)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        # train the model
        model.train()
        error = 0
        for input, target in train_loader:
            # print(f'Input shape: {input.shape}, Target shape: {target.shape}')
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            loss = model.train_loss(model(input), target)
            loss.backward()
            error += loss.item()

            optimizer.step()
            scheduler.step()

        if (epoch+1) % 20 == 0:
            # evaluate the model
            with torch.no_grad():
                model.eval()
                eval_error = 0
                total_samples = 0
                for input, target in eval_loader:
                    input, target = input.cuda(), target.cuda()
                    output = model(input)
                    loss = model.test_loss(output, target)
                    eval_error += loss.item()*len(input)
                    total_samples += len(input)

            # save the model
            torch.save(model.state_dict(), args.checkpoint)

            print(f'Epoch {epoch}, Train error: {error/len(train_loader)}, Eval error: {eval_error/total_samples}', flush=True)
        else:
            print(f'Epoch {epoch}, Train error: {error/len(train_loader)}', flush=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=0)
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default="foundation_model.pth")
    args = parser.parse_args()
    main(args)
