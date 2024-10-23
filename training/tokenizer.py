from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFC
from tokenizers.decoders import WordPiece
import os
import lightning as L
from itertools import chain
from pathlib import Path


class BPETokenizer:

    def __init__(
        self,
        out_dir: str = f"{os.getcwd()}/outputs",
        dataset: L.LightningDataModule = None,
    ):
        model_path = f"{out_dir}/bpe_tokenizer.json"

        self.tokenizer = (
            Tokenizer.from_file(model_path)
            if os.path.exists(model_path)
            else BPETokenizer._train(model_path, dataset)
        )

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def encode_batch(self, *args, **kwargs):
        return self.tokenizer.encode_batch(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def decode_batch(self, *args, **kwargs):
        return self.tokenizer.decode_batch(*args, **kwargs)

    def _train(model_path, dataset: L.LightningDataModule):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[SEP]", "[PAD]", "[MASK]"],
            continuing_subword_prefix="##",
            vocab_size=10000,
            max_token_length=10,
        )
        tokenizer.enable_padding()
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = NFC()
        tokenizer.decoder = WordPiece()
        tokenizer.train_from_iterator(
            chain.from_iterable(item for item in dataset.train_dataloader()),
            trainer=trainer,
        )

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        tokenizer.save(model_path)

        return tokenizer
