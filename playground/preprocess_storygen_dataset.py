from datasets import load_dataset, DatasetDict
import random
import os

ds = load_dataset("Dans-DiscountModels/RUCAIBox-Story-Generation-Alpaca")['train']

# merge 'instruction' and 'input' into 'prompt'
ds = ds.map(lambda x: {"prompt": x["instruction"] + "\n" + x["input"]})
ds = ds.remove_columns(["instruction", "input"])

# randomly select 100 rows
test_idx = random.sample(range(len(ds)), 100)
train_idx = [i for i in range(len(ds)) if i not in test_idx]

test_ds = ds.select(test_idx)
train_ds = ds.select(train_idx)

dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds
})

dataset_dict.push_to_hub("friendshipkim/RUCAIBox-Story-Generation-test", token=os.environ["HF_TOKEN"])

breakpoint()