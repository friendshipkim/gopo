from datasets import load_dataset, DatasetDict
import random
import os

print("Loading dataset...")
train_ds = load_dataset("HuggingFaceH4/ultrachat_200k")['train_sft']
test_ds = load_dataset("HuggingFaceH4/ultrachat_200k")['test_sft']

# delete messages column
train_ds = train_ds.remove_columns(["messages"])
test_ds = test_ds.remove_columns(["messages"])

# sample 1000 from train_sft
print("Sampling 500 from train_sft...")
val_idx = random.sample(range(len(train_ds)), 500)
train_idx = [i for i in range(len(train_ds)) if i not in val_idx]
val_ds = train_ds.select(val_idx)
train_ds = train_ds.select(train_idx)

print("Creating dataset dict...")
dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds,
    "val": val_ds
})
print(dataset_dict)

print("Pushing to hub...")
dataset_dict.push_to_hub(f"{os.environ['HF_USERNAME']}/UltraChat-200k", token=os.environ["HF_TOKEN"])

breakpoint()

