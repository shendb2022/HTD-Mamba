import torch
from torch import nn
from PyramidSSM import Mamba

'''
This part for designing the model to extract discriminative features
'''


class SpectralGroupAttention(nn.Module):
    def __init__(self, band=189, group_length=20, channel_dim=128, state_size=128, device=None, layer=1):
        super().__init__()
        stride = group_length // 4
        self.group_division = nn.Sequential(
            nn.Conv1d(1, channel_dim, group_length, stride=stride, padding=0),
            nn.LeakyReLU()
        )
        sequence_length = (band - group_length) // stride + 1
        self.mamba = Mamba(seq_len=sequence_length, d_model=channel_dim, state_size=state_size,
                           layer=layer)
        self.fnn = nn.Sequential(
            nn.Linear(sequence_length * channel_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.group_division(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.mamba(x)
        x = x.view([x.shape[0], -1])
        x = self.fnn(x)
        return x


if __name__ == '__main__':
    batch_size = 80
    bands = [189, 189, 205, 102]
    group_lengths = [30, 5, 5, 15]
    channels = [16, 16, 16, 16]
    num = 0
    input_size = (1, bands[num])
    model = SpectralGroupAttention(band=bands[num], group_length=group_lengths[num], channel_dim=channels[num],
                                   state_size=16,
                                   device="cuda:0", layer=1).cuda()

    from torchsummary import summary
    from thop import profile

    summary(model, input_size)

    # Calculate FLOPs and parameter count using thop
    input_tensor = torch.randn(batch_size, *input_size).cuda()
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    flops_million = flops / 10 ** 9
    params_million = params / 10 ** 6

    print(f'FLOPs: {flops_million:.6f} GFLOPs')
    print(f'Parameters: {params_million:.6f} MParams')
