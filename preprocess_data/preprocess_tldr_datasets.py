from datasets import load_dataset, DatasetDict
import random
import os

print("Loading dataset...")
ds = load_dataset("trl-lib/tldr")
train_ds = ds['train']
val_ds = ds['validation']
test_ds = ds['test']
print(ds)

# sample 1000 from train_sft
print("Sampling 500 from val...")
val_idx = random.sample(range(len(val_ds)), 500)
val_ds = val_ds.select(val_idx)

print("Creating dataset dict...")
dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds,
    "val": val_ds
})
print(dataset_dict)

print("Pushing to hub...")
dataset_dict.push_to_hub(f"{os.environ['HF_USERNAME']}/tldr", token=os.environ["HF_TOKEN"])

breakpoint()

