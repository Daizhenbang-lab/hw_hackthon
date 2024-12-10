import numpy as np
import zipfile
import os


def cache_H_data(cfg_path, input_data_path, output_data_path):    
    # read cfg file
    info = np.loadtxt(cfg_path, skiprows=1)
    info = info.astype(int)
    _, _, port_num, ant_num, sc_num = tuple(info.tolist())

    H = np.loadtxt(input_data_path)

    H = np.reshape(H, (-1, 2, sc_num, ant_num, port_num)) # n*2*408*64*2

    # stack real and imaginary parts into a dimension
    H_real = H[:, 0, :, :, :].transpose(0, 3, 2, 1)
    H_imag = H[:, 1, :, :, :].transpose(0, 3, 2, 1)

    H = np.stack((H_real, H_imag), axis=-1)

    H = H.astype(np.float16)

    np.save(output_data_path, H)


if __name__ == '__main__':
    dataset_idx = 0
    data_indices = [3]

    for data_idx in data_indices:
        zip_file = f'data/Dataset{dataset_idx}InputData{data_idx}.zip'
        cfg_file = f'data/Dataset{dataset_idx}CfgData{data_idx}.txt'

        if dataset_idx == 0:
            raw_file = f'data/Round{dataset_idx}InputData{data_idx}.txt'
        elif dataset_idx == 1:
            raw_file = f'data/Dataset{dataset_idx}InputData{data_idx}.txt'
        elif dataset_idx == 2:
            raw_file = f'data/Round{dataset_idx+1}InputData{data_idx}.txt'

        npy_file = f'data/Hf{dataset_idx}_{data_idx}.npy'

        # unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data/')

        print(f'Unzipped {zip_file}')

        # cache the data
        cache_H_data(cfg_file, raw_file, npy_file)

        print(f'Cached {raw_file} to {npy_file}')

        # remove the unzipped file
        os.remove(raw_file)
        print(f'Removed {raw_file}')
