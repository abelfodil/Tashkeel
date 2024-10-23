from datasets.tashkeela import TashkeelaDataModule
from tokenizer import BPETokenizer

data_module = TashkeelaDataModule()

data_module.prepare_data()
data_module.setup()

tokenizer = BPETokenizer(dataset=data_module)

train_dataloader = data_module.train_dataloader()

input_sentence_batch, target_sentence_batch = next(iter(train_dataloader))

print(input_sentence_batch)
print(target_sentence_batch)
