import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch


def standard(x):
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return np.zeros_like(x)
    return (x - min_value) / (max_value - min_value)


def cosin_similarity(x, y):
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y_multi = np.sum(np.multiply(x, y), axis=1)
    return x_y_multi / (x_norm * y_norm + 1e-8)


def patch_encoded(patch, center):
    p_h, p_w, b = patch.shape
    patch_unfold = np.reshape(patch, [-1, b], order='F')
    assert patch_unfold.shape[1] == center.shape[1]
    encoded_weight = cosin_similarity(patch_unfold, center)
    encoded_weight = np.exp(encoded_weight) / np.sum(np.exp(encoded_weight))
    encoded_weight = encoded_weight[:, None]
    encoded_vector = np.sum(encoded_weight * patch_unfold, axis=0)
    encoded_vector = encoded_vector[None, :]
    return encoded_vector


class Data(data.Dataset):
    def __init__(self, path, w_size=7):
        self.w_size = w_size
        self.pad_size = w_size // 2
        mat = sio.loadmat(path)
        img = mat['data']

        self.h, self.w, b = img.shape
        self.nums = self.h * self.w
        img = standard(img)

        self.data = np.pad(img, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
                           mode='reflect')

    def __getitem__(self, index):
        position_y = index // self.h
        position_x = index - position_y * self.h
        position_x = position_x + self.pad_size
        position_y = position_y + self.pad_size
        windows_out = self.data[position_x - self.pad_size:position_x + self.pad_size + 1,
                      position_y - self.pad_size:position_y + self.pad_size + 1, :]
        center = windows_out[self.pad_size, self.pad_size]
        center = center[None, :]
        patch = windows_out
        coded_vector = patch_encoded(patch, center)
        return torch.from_numpy(center).float(), torch.from_numpy(np.array(coded_vector)).float()

    def __len__(self):
        return self.nums


if __name__ == '__main__':
    data = Data('datasets/Sandiego.mat')
    center, coded_vector = data.__getitem__(128)
    plt.plot(center.T)
    plt.plot(coded_vector.T)
    plt.show()
