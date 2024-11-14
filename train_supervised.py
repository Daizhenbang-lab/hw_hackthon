from datasets_ import load_labelled_data
from models import ResNet18
from models.UNet import UNet
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import argparse
from tqdm import tqdm

# set gpu device
def main(args):
    torch.cuda.set_device(args.gpu)

    dataset_idx = args.dataset_idx
    data_idx = args.data_idx
    #C:/Users/代振邦/PycharmProjects/hw_hackthon/
    # Load the dataset
    train_set = load_labelled_data(
        f'data/H{dataset_idx}_{data_idx}.npy',
        f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')

    #model = ResNet18().cuda()
    model = UNet().cuda()

    num_epochs = args.num_epochs

    # build the dataloader and optimizer
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr=1e-, momentum=0.9, weight_decay=1e-4)


    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*num_epochs, eta_min=1e-5)

    # train the model
    for epoch in tqdm(range(num_epochs),'training epoch:'):
        model.train()
        error = 0
        for input, target in train_loader:
            input, target = input.cuda(), target.cuda()
            # add purturbation to improve the generalization
            noise = torch.randn_like(input)*2e-2
            input += noise
            optimizer.zero_grad()
            loss = model.train_loss(model(input), target)
            loss.backward()

            # log the error
            error += loss.item()

            optimizer.step()
            scheduler.step()

        print(f'Epoch {epoch}, Train error: {error/len(train_loader)}', flush=True)
        # save the model
        torch.save(model.state_dict(), f'model{dataset_idx}_{data_idx}.pth')



# double data
def main_(args):
    torch.cuda.set_device(args.gpu)
    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    train_set = load_labelled_data(
        f'data/H{dataset_idx}_{data_idx}.npy',
        f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')

    model = UNet().cuda()
    num_epochs = args.num_epochs


    original_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    original_data = []
    original_targets = []
    noisy_data = []
    noisy_targets = []
    for input, target in original_loader:
        original_data.append(input)
        original_targets.append(target)

        noise = torch.randn_like(input) * 2e-1
        input_noisy = input + noise
        noisy_data.append(input_noisy)
        noisy_targets.append(target)


    combined_data = torch.cat(original_data + noisy_data, dim=0)
    combined_targets = torch.cat(original_targets + noisy_targets, dim=0)


    combined_dataset = torch.utils.data.TensorDataset(combined_data, combined_targets)
    combined_train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)


    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(combined_train_loader) * num_epochs, eta_min=1e-5)


    for epoch in tqdm(range(num_epochs), 'training epoch:'):
        model.train()
        error = 0
        for input, target in combined_train_loader:
            input, target = input.cuda(), target.cuda()

            optimizer.zero_grad()
            loss = model.train_loss(model(input), target)
            loss.backward()

            error += loss.item()

            optimizer.step()
            scheduler.step()

        print(f'Epoch {epoch}, Train error: {error / len(combined_train_loader)}', flush=True)

        torch.save(model.state_dict(), f'model{dataset_idx}_{data_idx}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=0)
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()
    main_(args)
