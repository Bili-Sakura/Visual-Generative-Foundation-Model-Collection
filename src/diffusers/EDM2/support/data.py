import json
import os
import zipfile
from typing import Optional

import numpy as np
import PIL.Image
import torch

try:
    import pyspng
except ImportError:  # pragma: no cover
    pyspng = None


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, use_labels: bool = True, max_size: Optional[int] = None, random_seed: int = 0):
        self.path = path
        self.use_labels = use_labels
        self._zipfile = None
        if os.path.isdir(path):
            self.source_type = "dir"
            self.all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=path)
                for root, _dirs, files in os.walk(path)
                for fname in files
            }
        elif os.path.splitext(path)[1].lower() == ".zip":
            self.source_type = "zip"
            self.all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to directory or zip")

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {".npy"}
        self.image_fnames = sorted(fname for fname in self.all_fnames if os.path.splitext(fname)[1].lower() in supported_ext)
        if len(self.image_fnames) == 0:
            raise IOError("No images found")

        self.indices = np.arange(len(self.image_fnames), dtype=np.int64)
        if max_size is not None and len(self.indices) > max_size:
            rnd = np.random.RandomState(random_seed % (1 << 31))
            rnd.shuffle(self.indices)
            self.indices = np.sort(self.indices[:max_size])

        self._raw_labels = None
        sample = self._load_raw_image(self.indices[0])
        self.num_channels = int(sample.shape[0])
        self.resolution = int(sample.shape[1])

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.path)
        return self._zipfile

    def _open_file(self, fname: str):
        if self.source_type == "dir":
            return open(os.path.join(self.path, fname), "rb")
        return self._get_zipfile().open(fname, "r")

    def _load_raw_image(self, raw_idx: int):
        fname = self.image_fnames[raw_idx]
        ext = os.path.splitext(fname)[1].lower()
        with self._open_file(fname) as f:
            if ext == ".npy":
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == ".png" and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        if not self.use_labels or "dataset.json" not in self.all_fnames:
            return np.zeros([len(self.image_fnames), 0], dtype=np.float32)
        with self._open_file("dataset.json") as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return np.zeros([len(self.image_fnames), 0], dtype=np.float32)
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self.image_fnames]
        labels = np.array(labels)
        if labels.ndim == 1:
            one_hot = np.zeros((labels.shape[0], int(np.max(labels)) + 1), dtype=np.float32)
            one_hot[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1
            return one_hot
        return labels.astype(np.float32)

    @property
    def label_dim(self) -> int:
        return int(self._get_labels().shape[1])

    def _get_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels()
        return self._raw_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_idx = int(self.indices[idx])
        image = self._load_raw_image(raw_idx)
        label = self._get_labels()[raw_idx]
        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy()).to(torch.float32)
