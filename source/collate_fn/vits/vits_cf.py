import torch
import torch.utils.data


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        # mel: [freq, length]
        # wav: [1, length]
        # ppg: [len, 1024]
        # pit: [len]
        # spk: [256]
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spe_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])
        spe_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        spe_padded = torch.FloatTensor(
            len(batch), batch[0][0].size(0), max_spe_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spe_padded.zero_()
        wav_padded.zero_()

        max_ppg_len = max([x[2].size(0) for x in batch])
        ppg_lengths = torch.FloatTensor(len(batch))
        ppg_padded = torch.FloatTensor(
            len(batch), max_ppg_len, batch[0][2].size(1))
        vec_padded = torch.FloatTensor(
            len(batch), max_ppg_len, batch[0][3].size(1))
        pit_padded = torch.FloatTensor(len(batch), max_ppg_len)
        ppg_padded.zero_()
        vec_padded.zero_()
        pit_padded.zero_()
        spk = torch.FloatTensor(len(batch), batch[0][5].size(0))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spe = row[0]
            spe_padded[i, :, : spe.size(1)] = spe
            spe_lengths[i] = spe.size(1)

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            ppg = row[2]
            ppg_padded[i, : ppg.size(0), :] = ppg
            ppg_lengths[i] = ppg.size(0)

            vec = row[3]
            vec_padded[i, : vec.size(0), :] = vec

            pit = row[4]
            pit_padded[i, : pit.size(0)] = pit

            spk[i] = row[5]
        # print(ppg_padded.shape)
        # print(ppg_lengths.shape)
        # print(pit_padded.shape)
        # print(spk.shape)
        # print(spe_padded.shape)
        # print(spe_lengths.shape)
        # print(wav_padded.shape)
        # print(wav_lengths.shape)
        return (
            ppg_padded,
            ppg_lengths,
            vec_padded,
            pit_padded,
            spk,
            spe_padded,
            spe_lengths,
            wav_padded,
            wav_lengths,
        )