import pathlib
from typing import Union
from torch.utils.data import Dataset
from datasets.transforms.diacritics import StripDiacritics
import nltk

nltk.download("punkt_tab")


class TextFileDataset(Dataset):

    def __init__(
        self, file_path: Union[str, pathlib.Path], transform=StripDiacritics()
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        self.sentences = nltk.sent_tokenize(file_content)

        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return (self.transform(sentence) for sentence in self.sentences)

    def __getitem__(self, index):
        return self.transform(self.sentences[index])
