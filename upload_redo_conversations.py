import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Path to the data folder
data_path = "/home/regularpooria/Projects/WildCode/results/WildChat-20260102T025928Z-1-001/WildChat"


# Function to load CSV
def load_csv(file_path):
    return Dataset.from_csv(file_path)


# Collect all CSV files
csv_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

from datasets import concatenate_datasets

# Collect all datasets
all_datasets = []
for file in csv_files:
    parts = file.split("/")
    token_limit = "1200" if "1200" in parts[-2] else "800"
    print(f"File: {file}, parts[-2]: {parts[-2]}, token_limit: {token_limit}")
    filename = parts[-1]
    if filename.startswith("original"):
        variant = "original"
    elif filename.startswith("Act as"):
        variant = "act_as_security_specialist"
    elif filename.startswith("From security"):
        variant = "from_security_standpoint"
    elif filename.startswith("Varies") or filename.startswith("varies"):
        variant = "varies_according_to_vulnerability"
    else:
        variant = "unknown"
    try:
        ds = load_csv(file)

        # Add common columns
        def add_columns(x):
            return {
                **x,
                'max_output_tokens': int(token_limit),
                'output_version': f'output_{token_limit}',
                'variant': variant,
                'prompt_variant': x.get('prompt_variant') or ('original' if variant == 'original' else ''),
                'security_instruction': x.get('security_instruction') or '',
            }

        ds = ds.map(add_columns)
        all_datasets.append(ds)
        print(f"Loaded {token_limit}_{variant}")
    except Exception as e:
        print(f"Failed to load {token_limit}_{variant}: {e}")

# Concatenate all
combined_dataset = concatenate_datasets(all_datasets)

print("Combined dataset:", combined_dataset)

# Push to HF
combined_dataset.push_to_hub("regularpooria/wildcode_conversation_redo")
