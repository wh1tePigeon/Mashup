from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.vits.generator.nsf import SourceModuleHnNSF
from source.model.vits.modules.speaker_adapter import SpeakerAdapter

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

LOG = getLogger(__name__)

LRELU_SLOPE = 0.1


def padDiff(x):
    return F.pad(
        F.pad(x, (0, 0, -1, 1), "constant", 0) - x, (0, 0, 0, -1), "constant", 0
    )


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)



class NSFHifiGANGenerator(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        self.adapter = SpeakerAdapter(hp.vits.spk_dim, hp.gen.upsample_input)
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=np.prod(hp.gen.upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=hp.data.sampling_rate)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(hp.gen.upsample_input,
                               hp.gen.upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if hp.gen.resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            c_cur = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        hp.gen.upsample_initial_channel // (2 ** i),
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2)
                )
            )
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    Conv1d(
                        1,
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        #self.cond = nn.Conv1d(hp["gin_channels"], hp.gen.upsample_initial_channel, 1)

    def forward(self, spk, x, f0):
        x = x + torch.randn_like(x)
        x = self.adapter(x, spk)
        # LOG.info(1,x.shape,f0.shape,f0[:, None].shape)
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        # LOG.info(2,f0.shape)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x * torch.tanh(F.softplus(x))
        #x = x + self.cond(g)
        # LOG.info(124,x.shape,har_source.shape)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # LOG.info(3,x.shape)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            # LOG.info(4,x_source.shape,har_source.shape,x.shape)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        LOG.info("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)