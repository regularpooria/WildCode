#!/usr/bin/env python3
"""
Upload buggy conversation redo files to Hugging Face Hub as a dataset.
Each file will be structured with model_name and variant (original or act_as_a_security_researcher).
"""

import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Configuration
DATASET_NAME = "regularpooria/buggy-conversation-redo"  # Change this to your desired dataset name
SOURCE_DIR = Path("buggy_conversation_redo")

def extract_model_info(filename):
    """
    Extract model name and variant from filename.
    
    Examples:
        - act_as_security_researcher_command_a.json -> (command_a, act_as_a_security_researcher)
        - original_gpt_oss_120b.json -> (gpt_oss_120b, original)
    """
    name = filename.replace(".json", "")
    
    if name.startswith("act_as_security_researcher_"):
        model_name = name.replace("act_as_security_researcher_", "")
        variant = "act_as_a_security_researcher"
    elif name.startswith("original_"):
        model_name = name.replace("original_", "")
        variant = "original"
    else:
        # Fallback
        model_name = name
        variant = "unknown"
    
    return model_name, variant

def load_and_structure_data():
    """
    Load all JSON files and structure them with model_name and variant.
    Returns a list of dictionaries ready for dataset creation.
    """
    all_data = []
    
    for json_file in SOURCE_DIR.glob("*.json"):
        print(f"Processing {json_file.name}...")
        
        model_name, variant = extract_model_info(json_file.name)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each programming language in the JSON
            for language, conversations in data.items():
                for conversation_id, conversation_data in conversations.items():
                    record = {
                        'model_name': model_name,
                        'variant': variant,
                        'language': language,
                        'conversation_id': conversation_id,
                        'user_prompt': conversation_data.get('USER_PROMPT', ''),
                        'response': conversation_data.get('RESPONSE', '')
                    }
                    all_data.append(record)
            
            print(f"  ✓ Loaded {len([k for v in data.values() for k in v.keys()])} conversations")
        
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
    
    return all_data

def create_dataset(data):
    """
    Create a Hugging Face Dataset from the structured data.
    """
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Print some statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total records: {len(dataset)}")
    print(f"\nRecords by variant:")
    for variant in dataset.unique('variant'):
        count = len([r for r in data if r['variant'] == variant])
        print(f"  - {variant}: {count}")
    
    print(f"\nRecords by model:")
    for model in sorted(dataset.unique('model_name')):
        count = len([r for r in data if r['model_name'] == model])
        print(f"  - {model}: {count}")
    
    print(f"\nLanguages covered:")
    for lang in sorted(dataset.unique('language')):
        count = len([r for r in data if r['language'] == lang])
        print(f"  - {lang}: {count}")
    
    return dataset

def main():
    """Main execution function."""
    print("="*60)
    print("Buggy Conversation Redo Dataset Uploader")
    print("="*60)
    
    # Check if source directory exists
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found!")
        return
    
    # Login to Hugging Face (will prompt for token if needed)
    print("\nLogging in to Hugging Face...")
    try:
        login()
    except Exception as e:
        print(f"Error logging in: {e}")
        print("Please make sure you have a valid Hugging Face token.")
        print("You can create one at: https://huggingface.co/settings/tokens")
        return
    
    # Load and structure data
    print("\n" + "="*60)
    print("Loading and structuring data...")
    print("="*60)
    data = load_and_structure_data()
    
    if not data:
        print("No data found! Please check your source directory.")
        return
    
    # Create dataset
    print("\n" + "="*60)
    print("Creating dataset...")
    print("="*60)
    dataset = create_dataset(data)
    
    # Push to hub
    print("\n" + "="*60)
    print(f"Uploading to Hugging Face Hub: {DATASET_NAME}")
    print("="*60)
    
    try:
        dataset.push_to_hub(
            DATASET_NAME,
            private=False,  # Set to True if you want a private dataset
            commit_message="Upload buggy conversation redo dataset with model names and variants"
        )
        print(f"\n✓ Successfully uploaded dataset to: https://huggingface.co/datasets/{DATASET_NAME}")
        
        # Create a README
        readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code
- security
- vulnerabilities
- conversations
size_categories:
- n<1K
---

# Buggy Conversation Redo Dataset

This dataset contains conversations about code generation with potential security vulnerabilities.
It includes responses from different models with two variants:
- **original**: Standard model responses
- **act_as_a_security_researcher**: Responses where the model was prompted to act as a security researcher

## Dataset Structure

Each record contains:
- `model_name`: The name of the model used (e.g., gpt_oss_120b, command_a, etc.)
- `variant`: Either "original" or "act_as_a_security_researcher"
- `language`: The programming language discussed
- `conversation_id`: Unique identifier for the conversation
- `user_prompt`: The user's input/question
- `response`: The model's response

## Models Included

{', '.join(sorted(set([r['model_name'] for r in data])))}

## Languages Covered

{', '.join(sorted(set([r['language'] for r in data])))}

## Statistics

- Total conversations: {len(dataset)}
- Original variant: {len([r for r in data if r['variant'] == 'original'])}
- Security researcher variant: {len([r for r in data if r['variant'] == 'act_as_a_security_researcher'])}

## Citation

If you use this dataset, please cite:
```bibtex
@dataset{{buggy_conversation_redo,
  author = {{Your Name}},
  title = {{Buggy Conversation Redo Dataset}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/{DATASET_NAME}}}
}}
```
"""
        
        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=DATASET_NAME,
            repo_type="dataset",
            commit_message="Add dataset README"
        )
        
        print("✓ README.md uploaded successfully")
        
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        return
    
    print("\n" + "="*60)
    print("Upload complete!")
    print("="*60)

if __name__ == "__main__":
    main()
