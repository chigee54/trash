from data_tool.data_utils import load_datasets, save_samples
from data_tool import SentencesDataset
import os

output_path = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

train_samples = load_datasets(datasets=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"], need_label=False,
                              use_all_unsupervised_texts=True, no_pair=True)

save_samples(train_samples, os.path.join(output_path, "train_texts.txt"))

train_dataset = SentencesDataset(train_samples, model)

# train_dataloader = DataLoader(train_dataset, shuffle=not args.no_shuffle, batch_size=train_batch_size)
