from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union
import torchaudio as ta
import torch
from torch import nn


from .utils import (MelBandsplitSpecification,
                    MusicalBandsplitSpecification,
                    VocalBandsplitSpecification,)

from .core import MultiSourceMultiMaskBandSplitCoreRNN


def get_band_specs(band_specs, n_fft, fs, n_bands=None):
    if band_specs in ["dnr:speech", "dnr:vox7", "musdb:vocals", "musdb:vox7"]:
        bsm = VocalBandsplitSpecification(
                nfft=n_fft, fs=fs
        ).get_band_specs()
        freq_weights = None
        overlapping_band = False
    elif "musical" in band_specs:
        assert n_bands is not None
        specs = MusicalBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif band_specs == "dnr:mel" or "mel" in band_specs:
        assert n_bands is not None
        specs = MelBandsplitSpecification(
                nfft=n_fft,
                fs=fs,
                n_bands=n_bands
        )
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    else:
        raise NameError

    return bsm, freq_weights, overlapping_band


class MultiMaskMultiSourceBandSplitRNN(nn.Module):
    def __init__(
            self,
            in_channel: int,
            stems: List[str],
            band_specs: Union[str, List[Tuple[float, float]]] = "musical",
            fs: int = 44100,
            require_no_overlap: bool = False,
            require_no_gap: bool = True,
            normalize_channel_independently: bool = False,
            treat_channel_as_feature: bool = True,
            n_sqm_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            cond_dim: int = 0,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            mlp_dim: int = 512,
            hidden_activation: str = "Tanh",
            hidden_activation_kwargs: Optional[Dict] = None,
            complex_mask: bool = True,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            n_bands: int = None,
            use_freq_weights: bool = True,
            normalize_input: bool = False,
            mult_add_mask: bool = False,
            freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        assert power is None

        window_fn = torch.__dict__[window_fn]

        if isinstance(band_specs, str):
            self.band_specs, self.freq_weights, self.overlapping_band = get_band_specs(
                band_specs,
                n_fft,
                fs,
                n_bands
                )
        self.stems = stems
        
        self.stft = (
                ta.transforms.Spectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        pad=0,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        power=power,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )

        self.istft = (
                ta.transforms.InverseSpectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        pad=0,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreRNN(
                stems=stems,
                band_specs=self.band_specs,
                in_channel=in_channel,
                require_no_overlap=require_no_overlap,
                require_no_gap=require_no_gap,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
                n_sqm_modules=n_sqm_modules,
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                hidden_activation=hidden_activation,
                hidden_activation_kwargs=hidden_activation_kwargs,
                complex_mask=complex_mask,
                overlapping_band=self.overlapping_band,
                freq_weights=self.freq_weights,
                n_freq=n_fft // 2 + 1,
                use_freq_weights=use_freq_weights,
                mult_add_mask=mult_add_mask
        )

        self.normalize_input = normalize_input
        self.cond_dim = cond_dim

        if freeze_encoder:
            for param in self.bsrnn.band_split.parameters():
                param.requires_grad = False

            for param in self.bsrnn.tf_model.parameters():
                param.requires_grad = False

    def forward(self, batch):
        # with torch.no_grad():
        audio = batch["audio"]
        cond = batch.get("condition", None)
        with torch.no_grad():
            batch["spectrogram"] = {stem: self.stft(audio[stem]) for stem in
                                    audio}

        X = batch["spectrogram"]["mixture"]
        length = batch["audio"]["mixture"].shape[-1]

        output = self.bsrnn(X, cond=cond)
        output["audio"] = {}

        for stem, S in output["spectrogram"].items():
            s = self.istft(S, length)
            output["audio"][stem] = s

        return batch, output
