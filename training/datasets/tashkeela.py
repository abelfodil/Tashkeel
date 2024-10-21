# Source: https://www.kaggle.com/datasets/linuxscout/tashkeela
# Reference: http://dx.doi.org/10.1016/j.dib.2017.01.011

import pathlib
import os.path
import urllib.request
import zipfile
import tarfile
import lightning as L
import glob
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
from datasets.textfile_dataset import TextFileDataset


class TashkeelaDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = f"{pathlib.Path(__file__).parent.resolve()}/downloads"
    ):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        dataset_zip_path = f"{self.data_dir}/tashkeela.tar.bz2.zip"
        dataset_path = f"{self.data_dir}/Tashkeela-arabic-diacritized-text-utf8-0.3"
        dataset_tar_path = f"{dataset_path}.tar.bz2"

        self.documents_paths = glob.iglob(
            f"{dataset_path}/texts.txt/**/*.txt", recursive=True
        )

        if os.path.exists(dataset_path):
            return

        if not os.path.exists(dataset_zip_path):
            urllib.request.urlretrieve(
                "https://www.kaggle.com/api/v1/datasets/download/linuxscout/tashkeela",
                dataset_zip_path,
            )

        if not os.path.exists(dataset_tar_path):
            with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

        if not os.path.exists(dataset_path):
            with tarfile.open(dataset_tar_path, "r:bz2") as tar_ref:
                tar_ref.extractall(self.data_dir, filter="tar")

    def setup(self, stage: str = None):
        dataset = ConcatDataset(
            TextFileDataset(document_path) for document_path in self.documents_paths
        )
        self.train, self.val, self.test, self.predict = random_split(
            dataset, [0.8, 0.1, 0.1, 0], generator=torch.Generator()
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=32)
