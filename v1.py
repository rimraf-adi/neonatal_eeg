class V_One :

    def __init__(self):
        import numpy as np
        import pandas as pd
        import itertools
        import mne
        import cupy as cp

        bipolar_pairs = [
        ('EEG Fp1-REF', 'EEG F7-REF'),
        ('EEG F7-REF',  'EEG T3-REF'),
        ('EEG T3-REF',  'EEG T5-REF'),
        ('EEG T5-REF',  'EEG O1-REF'),
        ('EEG Fp1-REF', 'EEG F3-REF'),
        ('EEG F3-REF',  'EEG C3-REF'),
        ('EEG C3-REF',  'EEG P3-REF'),
        ('EEG P3-REF',  'EEG O1-REF'),
        ('EEG Fz-REF',  'EEG Cz-REF'),
        ('EEG Cz-REF',  'EEG Pz-REF'),
        ('EEG Fp2-REF', 'EEG F4-REF'),
        ('EEG F4-REF',  'EEG C4-REF'),
        ('EEG C4-REF',  'EEG P4-REF'),
        ('EEG P4-REF',  'EEG O2-REF'),
        ('EEG Fp2-REF', 'EEG F8-REF'),
        ('EEG F8-REF',  'EEG T4-REF'),
        ('EEG T4-REF',  'EEG T6-REF'),
        ('EEG T6-REF',  'EEG O2-REF'),
    ]

    desired_order = [
        'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'Fz-Cz', 'Cz-Pz',
    ]

    def pair_name(pair):
        left = pair[0].replace('EEG ', '').replace('-REF', '')
        right = pair[1].replace('EEG ', '').replace('-REF', '')
        return f"{left}-{right}"

    name_to_pair = {pair_name(p): p for p in bipolar_pairs}

    reordered_pairs = [name_to_pair[name] for name in desired_order]

    for name, pair in zip(desired_order, reordered_pairs):
        print(f"{name}: {pair}")

    bipolar_pairs = [
        ('EEG Fp1-REF', 'EEG F7-REF'),
        ('EEG F7-REF',  'EEG T3-REF'),
        ('EEG T3-REF',  'EEG T5-REF'),
        ('EEG T5-REF',  'EEG O1-REF'),
        ('EEG Fp1-REF', 'EEG F3-REF'),
        ('EEG F3-REF',  'EEG C3-REF'),
        ('EEG C3-REF',  'EEG P3-REF'),
        ('EEG P3-REF',  'EEG O1-REF'),
        ('EEG Fz-REF',  'EEG Cz-REF'),
        ('EEG Cz-REF',  'EEG Pz-REF'),
        ('EEG Fp2-REF', 'EEG F4-REF'),
        ('EEG F4-REF',  'EEG C4-REF'),
        ('EEG C4-REF',  'EEG P4-REF'),
        ('EEG P4-REF',  'EEG O2-REF'),
        ('EEG Fp2-REF', 'EEG F8-REF'),
        ('EEG F8-REF',  'EEG T4-REF'),
        ('EEG T4-REF',  'EEG T6-REF'),
        ('EEG T6-REF',  'EEG O2-REF'),
    ]

    def normalize_channel(ch):
        return ch.strip().upper()

    bipolar_pairs = [(normalize_channel(a), normalize_channel(b)) for a, b in bipolar_pairs]

    def make_ch_names(pairs):
        def pretty(ch):
            return ch.replace('EEG ', '').replace('-REF', '').capitalize()
        return [f"{pretty(a)}-{pretty(b)}" for a, b in pairs]

    anode = [a for a, _ in bipolar_pairs]
    cathode = [b for _, b in bipolar_pairs]
    ch_names = make_ch_names(reordered_pairs)

    def getArray(filename: str):
        raw = mne.io.read_raw_edf(filename, preload=True)
        raw.rename_channels(lambda ch: ch.upper())
        drop_candidates = ['ECG EKG', 'RESP EFFORT','ECG EKG-REF','RESP EFFORT-REF']
        available = set(raw.ch_names)
        to_drop = [ch for ch in drop_candidates if ch in available]
        if to_drop:
            raw.drop_channels(to_drop)
        raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=ch_names, copy=True)
        epochs=mne.make_fixed_length_epochs(raw,duration=1,overlap=0)
        array = epochs.get_data()
        return cp.asarray(array)
    import os

    self.eeg_set = []
    for i in range(1, 80):
        filepath = f'./data/eeg{i}.edf'
        if os.path.exists(filepath):
            try:
                eeg = getArray(filepath)
                self.eeg_set.append(eeg)
            except ValueError as e:
                print(f"Error in file: {filepath}")
                print(e)

     