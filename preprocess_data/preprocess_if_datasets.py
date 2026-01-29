# load instruction following datasets and merge them

import os
from datasets import load_dataset, Value, DatasetDict
import random

train_dataset_name = "allenai/tulu-3-sft-personas-instruction-following"
test_dataset_name = "google/IFEval"

train_dataset = load_dataset(train_dataset_name)["train"]
test_dataset = load_dataset(test_dataset_name)["train"]
print("original datasets:")
print("train: ", train_dataset_name)
print(train_dataset)
print("test: ", test_dataset_name)
print(test_dataset)

# merge the datasets
# (Pdb) train_dataset.column_names
# ['id', 'prompt', 'messages', 'constraints']
# (Pdb) test_dataset.column_names
# ['key', 'prompt', 'instruction_id_list', 'kwargs']

# convert data type of key to string
test_dataset = test_dataset.cast_column("key", Value('string'))

# rename columns before merging
test_dataset = test_dataset.rename_column("key", "id")
test_dataset = test_dataset.rename_column("instruction_id_list", "instruction_id_list_ifeval")
test_dataset = test_dataset.rename_column("kwargs", "kwargs_ifeval")


# add empty columns with None values to allow proper casting
# For train dataset: add instruction_id_list_ifeval and kwargs_ifeval 
train_dataset = train_dataset.add_column("instruction_id_list_ifeval", [None for _ in range(len(train_dataset))])
train_dataset = train_dataset.add_column("kwargs_ifeval", [None for _ in range(len(train_dataset))])

# For test dataset: add messages and constraints 
test_dataset = test_dataset.add_column("messages", [None for _ in range(len(test_dataset))])
test_dataset = test_dataset.add_column("constraints", [None for _ in range(len(test_dataset))])

# Cast columns to ensure matching types
from datasets import Features, Sequence, Value
train_features = train_dataset.features
test_features = test_dataset.features

# Create unified features schema
unified_features = Features({
    'id': Value('string'),
    'prompt': Value('string'), 
    'messages': train_features['messages'],  # Use train's messages structure
    'constraints': train_features['constraints'],  # Use train's constraints structure
    'instruction_id_list_ifeval': test_features['instruction_id_list_ifeval'],  # Use test's structure
    'kwargs_ifeval': test_features['kwargs_ifeval'],  # Use test's structure
    'dataset_source': Value('string')
})

# add dataset source before casting
train_dataset = train_dataset.add_column("dataset_source", [train_dataset_name for _ in range(len(train_dataset))])
test_dataset = test_dataset.add_column("dataset_source", [test_dataset_name for _ in range(len(test_dataset))])

# Cast both datasets to use the unified schema
train_dataset = train_dataset.cast(unified_features)
test_dataset = test_dataset.cast(unified_features)

# match the column order of train_dataset and test_dataset
column_order = ['id', 'prompt', 'messages', 'constraints', 'instruction_id_list_ifeval', 'kwargs_ifeval', 'dataset_source']
train_dataset = train_dataset.select_columns(column_order)
test_dataset = test_dataset.select_columns(column_order)

# sample 500 from train_dataset
print("Sampling 500 from train_dataset...")
val_idx = random.sample(range(len(train_dataset)), 500)
train_idx = [i for i in range(len(train_dataset)) if i not in val_idx]
val_dataset = train_dataset.select(val_idx)
train_dataset = train_dataset.select(train_idx)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
})
print("processed dataset:")
print(dataset_dict)

# push to hub
# make sure to set HF_TOKEN in the environment variables
dataset_dict.push_to_hub(f"{os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval", token=os.environ["HF_TOKEN"])