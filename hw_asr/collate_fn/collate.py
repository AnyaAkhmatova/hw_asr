import logging
from typing import List

import torch
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    bs = len(dataset_items)
    # "audio", "spectrogram", "duration", "text", "text_encoded", "audio_path"

    audio_max_len = np.array([item['audio'].shape[1] for item in dataset_items]).max()
    audios = torch.zeros(bs, audio_max_len)
    audios_lengths = torch.zeros(bs, dtype=torch.int)

    spec_n_mels = dataset_items[0]['spectrogram'].shape[1]
    spec_max_len = np.array([item['spectrogram'].shape[2] for item in dataset_items]).max()
    specs = torch.zeros(bs, spec_n_mels, spec_max_len)
    specs_lengths = torch.zeros(bs, dtype=torch.int)

    durations = torch.zeros(bs)

    texts = []
    
    text_enc_max_len = np.array([item['text_encoded'].shape[1] for item in dataset_items]).max()
    text_encs = torch.zeros(bs, text_enc_max_len)
    text_encs_lengths = torch.zeros(bs, dtype=torch.int)

    audio_paths = []    
    
    for i, item in enumerate(dataset_items):
        audios[i, :item['audio'].shape[1]] = item['audio']
        audios_lengths[i] = item['audio'].shape[1]

        specs[i, :, :item['spectrogram'].shape[2]] = item['spectrogram']
        specs_lengths[i] = item['spectrogram'].shape[2]

        durations[i] = item['duration']

        texts.append(item['text'])

        text_encs[i, :item['text_encoded'].shape[1]] = item['text_encoded']
        text_encs_lengths[i] = item['text_encoded'].shape[1]

        audio_paths.append(item['audio_path'])
        
    result_batch['audio'] = audios
    result_batch['audio_length'] = audios_lengths

    result_batch['spectrogram'] = specs
    result_batch['spectrogram_length'] = specs_lengths

    result_batch['duration'] = durations

    result_batch['text'] = np.array(texts)

    result_batch['text_encoded'] = text_encs
    result_batch['text_encoded_length'] = text_encs_lengths

    result_batch['audio_path'] = np.array(audio_paths)
    
    return result_batch

