# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import os
import random
from typing import Tuple
from pathlib import Path

import torch
from torchvision.transforms import Compose
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS, HASH_DIVIDER, EXCEPT_FOLDER, _load_list
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio import transforms, load
import flwr #! DON'T REMOVE -- bad things happen


# [1]: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
# [2] HelloEdge: Keyword Spotting on Microcontrollers (https://arxiv.org/abs/1711.07128)
# [3] Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition (https://arxiv.org/abs/1804.03209)

BKG_NOISE = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav", "running_tap.wav", "white_noise.wav"]

CLASSES_12 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

def pad_sequence(batch): #! borrowed from [1]
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def label_to_index(word, labels): #! borrowed from [1]
    # Return the position of the word in labels
    if word in labels:
        return torch.tensor(labels.index(word))
    else:
        return torch.tensor(10) # higlight as `unknown`


class PartitionedSPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, data_path: Path, subset: str, transforms: list, classes: str = 'all'):
        '''classes:
        # v1: either 30 (all) or 12 (10 + unknown + silence)
        # v2: either 35 (all) or 12 (10 + unknown + silence)
        '''

        super().__init__(data_path, url="",
                         folder_in_archive="",
                         subset=subset, download=False)

        self.subset = subset
        self.transforms = transforms
        self.device = 'cpu'

        cls = [Path(f.path).name for f in os.scandir(self._path) if f.is_dir() and f.path != str(Path(self._path)/EXCEPT_FOLDER)]
        if "federated" in cls:
            cls.remove("federated") # if data_path points to the whole dataset (i.e. not inside /federated), we'll hit this

        self.classes_to_use = cls if classes=='all' else CLASSES_12
        # self.collate_fn = get_collate(self.classes_to_use, self.transforms)

        # let's pre-load all background audio clips. This should help when
        # blending keyword audio with bckg noise
        self.background_sounds = []
        for noise in BKG_NOISE:
            path = data_path/EXCEPT_FOLDER/noise
            path = os.readlink(path) if os.path.islink(path) else path
            waveform, sample_rate = load(path)
            self.background_sounds.append([waveform, sample_rate])

        # now let's assume we have 10% more data representing `silence`.
        #! Hack alert: we artificially add paths (that do not exist) to the _walker.
        #! When this path is chosen via __getitem__, it will be detected as w/ label "silence"and the file itself wont' be loaded. Instead a silence clip (i.e. all zeros) will be returned
        #! Silence support is done in self._load_speechcommands_item_with_silence_support()
        if 'silence' in self.classes_to_use:
            # append silences to walker in dataset object
            for _ in range(int(len(self._walker)*0.1)):
                self._walker.append(data_path/"silence/sdfsfdsf.wav")

        # print(f"Dataset contains {len(self._walker)} audio files")


    def _collate_fn(self, batch): #! ~borrowed from [1]
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [label_to_index(label, self.classes_to_use)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        # tensors = tensors.to(self.device)
        tensor_t = self.transforms(tensors)

        return tensor_t, targets

    def _get_labels_histogram(self):
        """returns histogram of labels"""
        hist = [0] * len(self.classes_to_use)
        for p in self._walker:
            path = os.readlink(p) if os.path.islink(p) else p
            label = Path(path).parent.name
            hist[label_to_index(label, self.classes_to_use)] += 1
        return hist

    def get_balanced_sampler(self):
        """This construct a [1,N] array w/ N the number of datapoints in the datasets. Each
        gets assigned a probabily of being added to a batch of data. This will be passed to a initialise
        a WeightedRandomSampler and return it."""

        hist = self._get_labels_histogram()
        weight_per_class = [len(self._walker)/float(count) if count>0 else 0 for count in hist]
        w = [0] * len(self._walker)
        for i, p in enumerate(self._walker):
            path = os.readlink(p) if os.path.islink(p) else p
            label = Path(path).parent.name
            label_idx = label_to_index(label, self.classes_to_use)
            w[i] = weight_per_class[label_idx]

        sampler = WeightedRandomSampler(w, len(w))
        return sampler, hist

    def _decode_classes(self, labels: torch.tensor):
        return [self.classes_to_use[i] for i in labels]


    def _extract_from_waveform(self, waveform, sample_rate):
        """Returns a waveform of `sample_rate` samples of the
        inputed `waveform`. If `sample_rate` is that of the `waveform`
        then the returned waveform will be 1s long."""
        min_t = 0
        max_t = waveform.shape[1] - sample_rate
        off_set = random.randint(min_t, max_t)
        return waveform[:, off_set:off_set+sample_rate]


    def _load_speechcommands_item_with_silence_support(self, filepath:str, path: str):
        """if loading `silence` we extract a 1s random clip from the background audio
        files in SpeechCommands dataset (this is how the `silence` category should be
        constructed according to the SpeechCommands paper). Else, the
        behaviour is the same as in the default SPECHCOMMANDS dataset"""
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)

        if label == 'silence':
            # construct path to a random .wav in background_noise dir
            # filepath = path + '/' + EXCEPT_FOLDER + "/" + random.sample(BKG_NOISE,1)[0]
            # picking one random pre-loaded background sound
            waveform, sample_rate = random.sample(self.background_sounds,1)[0]

            # let's extact a 1s sequence
            waveform = self._extract_from_waveform(waveform, sample_rate)
            utterance_number = -1
            speaker_id = -1
        else:

            speaker, _ = os.path.splitext(filename)
            speaker, _ = os.path.splitext(speaker)

            speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
            utterance_number = int(utterance_number)

            # Load audio
            # print(f"loading: {filepath}")
            waveform, sample_rate = load(filepath)

        return waveform, sample_rate, label, speaker_id, utterance_number


    def _apply_time_shift(self, waveform, sample_rate):
        """Applies time shifting (positive or negative). Hardcoded
        to apply rand(-100ms, +100ms)."""

        # apply random time shift of [-100ms, 100ms]
        shift_ammount = sample_rate/10 # this will give us a 10th of a 1s signal
        shift = random.randint(-shift_ammount, shift_ammount)
        if shift < 0:
            waveform = waveform[:, abs(shift):] # will be padded with zeros later on in collate_fn
        else:
            waveform_ = torch.zeros_like(waveform)
            waveform_[:, shift:] = waveform[:, :waveform.shape[1]-shift]
            waveform = waveform_

        return waveform


    def _blend_with_background(self, waveform):

        background_volume = 0.1 #  the default in [2] #! this seems to limit acc -- maybe lower is better?
        background_frequency = 0.8 # ratio of samples that will get background added in (as in [2])

        if random.uniform(0.0, 1.0) < background_frequency:
            volume = random.uniform(0.0, background_volume) # as in [2]
            noise, _ = random.sample(self.background_sounds,1)[0]
            noise = self._extract_from_waveform(noise, waveform.shape[1])
            return (1.0 - volume)*waveform + volume*noise
        else:
            return waveform


    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, int]:
        fileid = self._walker[n]
        path = self._path

        if os.path.islink(fileid):
            fileid = os.readlink(fileid)
            path = Path(self._path).parent.parent

        wvfrm, sr, label, speaker_id, utt_num = self._load_speechcommands_item_with_silence_support(fileid, path)

        if self.subset == "training":
            wvfrm = self._apply_time_shift(wvfrm, sr)
            wvfrm = self._blend_with_background(wvfrm)

        return wvfrm, sr, label, speaker_id, utt_num


def get_speechcommands_and_partition_it(destination_path: Path, version: int):
    """Downloads SpeechCommands dataset if not found and partitions it by
    `session ID` (which is a randomly generated alphanumeric sequence prefixing
    each audio file and can be use as speaker identifier -- according to [3]).
    Dataset statistics:
        v1: 64721 .wav files from 1881 speakers (1503 for training)
        v2: 105829 .wav files from 2618 speakers (2112 for training)
    """

    assert version in [1,2], f"Only version `1` or `2` are understood. You chose: {version}"

    path = Path(destination_path)
    path.mkdir(exist_ok=True)
    url = f"speech_commands_v0.0{version}"
    folder_in_archive = "SpeechCommands"
    whole_dataset = SPEECHCOMMANDS(path, url=url, folder_in_archive=folder_in_archive, subset=None, download=True)

    # get class all classes names
    cls_names = [Path(f.path).name for f in os.scandir(whole_dataset._path) if f.is_dir() and f.path != str(Path(whole_dataset._path)/EXCEPT_FOLDER)]

    if "federated" in cls_names:
        cls_names.remove("federated")

    # now we generate the `federated` directory
    fed_dir = Path(whole_dataset._path)/"federated"
    if not fed_dir.exists():
        fed_dir.mkdir()

        print(f"{len(cls_names)} (total) classes found")
        print(f"Dataset has: {len(whole_dataset._walker)} .wav files")

        # Get speakers IDs
        unique_ids = []
        for wav in whole_dataset._walker:
            wav = Path(wav)
            session_id = wav.stem[:wav.stem.find(HASH_DIVIDER)]
            if session_id not in unique_ids:
                unique_ids.append(session_id)

        print(f"Unique speaker IDs found: {len(unique_ids)}")

        # From all the IDs, some are **excluselively** in the test set, others exclusively in
        # the validation set and the rest form the training set. Now we identify which
        # belongs to which split.

        val_list = _load_list(whole_dataset._path, "validation_list.txt")
        test_list = _load_list(whole_dataset._path, "testing_list.txt")
        train_ids = []
        val_ids = []
        test_ids = []
        for i, id in enumerate(unique_ids):
            for wav in whole_dataset._walker:
                if id in wav:
                    if wav in val_list:
                        val_ids.append(id)
                    elif wav in test_list:
                        test_ids.append(id)
                    else:
                        train_ids.append(id)
                    break

        print(f"Clients for training ({len(train_ids)}), validation ({len(val_ids)}), testing ({len(test_ids)})")

        assert len(train_ids)+len(val_ids)+len(test_ids) == len(unique_ids), "This shouldn't happen"

        # partition dataset, creating a directory for each speaker id in TRAINING SET
        # we create symlinks to all the files with the same ID and place
        # them in their corresponding lable directory
        # Then we create `testing_list.txt` and `validation_list.txt` so we can reuse
        # the logic in the torchvision.SPEECHCOMMANDS logic. These will be empty

        # get list of files for training (tehste lines are borrowed from SpeechCommands dataset class)
        excludes = set(_load_list(whole_dataset._path, "validation_list.txt", "testing_list.txt"))
        walker = sorted(str(p) for p in Path(whole_dataset._path).glob("*/*.wav"))
        train_files = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]

        print(f"Creating training paritions. Creating symlinks to training data...")
        for i, id in enumerate(train_ids):
            client_dir = Path(fed_dir/str(i))
            client_dir.mkdir()

            # ensure all classes have a directory (this will be relevant for PartitionedSPEECHCOMMANDS as it will
            # be required to figureout the classes in the dataset)
            for cls in cls_names:
                (client_dir/str(cls)).mkdir()

            # create empyt `testing_list.txt` and `validation_list.txt`
            (client_dir/"testing_list.txt").touch()
            (client_dir/"validation_list.txt").touch()

            for file in train_files:
                if id in file: # if file belongs to this speaker ID
                    cls = Path(file).parent.stem
                    os.symlink(file, client_dir/cls/Path(file).name)

            # symlink also background sounds
            (client_dir/EXCEPT_FOLDER).mkdir()
            for each_file in (Path(whole_dataset._path)/EXCEPT_FOLDER).glob('*.wav*'):
                os.symlink(each_file, client_dir/EXCEPT_FOLDER/each_file.name)


        print("Done")

    return fed_dir


def raw_audio_to_mfcc_transforms():
    # values from [1], [2]: Here we transform the raw audio wave into MFCC features
    # which encode each audio clip into a 2D matrix.
    # This allows us to treat audio signals as images
    ss = 8000 # 8KHz
    n_mfcc = 40
    window_width = 40e-3 # length of window in seconds
    stride = 20e-3 # stride between windows
    n_fft = 400
    T = Compose([transforms.Resample(16000, ss),
                 transforms.MFCC(sample_rate=ss,
                                 n_mfcc=n_mfcc,
                                 melkwargs={'win_length': int(ss*window_width),
                                 'hop_length': int(ss*stride),
                                 'n_fft': n_fft}
                                 )
                ])
    return T


def test(fed_dir: str, client_id: int, device: str, test_whole_dataset: bool = False):
    '''Loads dataset for clien with id=client_id'''

    if test_whole_dataset:
        data_path = Path(fed_dir).parent
    else:
        data_path = Path(fed_dir)/str(client_id)
    dataset = PartitionedSPEECHCOMMANDS(data_path, "training", transforms=raw_audio_to_mfcc_transforms(), classes='all')

    # if not using the whole labels provided (i.e. classes="all"), SpeechCommands is very inbalanced since many labels are collapsed under the "unknown" label
    sampler, hist = dataset.get_balanced_sampler()
    print("Histogram of labels:")
    print(hist)

    train_loader = DataLoader(dataset, batch_size=20, shuffle=False,
                      num_workers=6, collate_fn=dataset._collate_fn,
                      sampler=sampler, pin_memory=True)

    for clip, lbl in train_loader:
        clip = clip.to(device)
        print(f"batch shape: {clip.shape}, {clip.type()}")
        lbl = lbl.to(device)
        print(f"labels: {lbl} ---> {dataset._decode_classes(lbl)}")


if __name__ == "__main__":

    version = 2
    fed_dir = get_speechcommands_and_partition_it('./data', version=version)
    print(f"{fed_dir = }")

    test(fed_dir, client_id=99, device='cuda', test_whole_dataset=True)
