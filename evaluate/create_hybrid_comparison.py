#!/usr/bin/env python3
"""
Script to create hybrid comparison JSON files by combining completions from two different JSON files.

This script allows you to:
1. Select 2 JSON files from completion_final/
2. Choose which model path (and corresponding completion) to take from each file
3. Create a new JSON file with the selected completions
4. Hard stop if there are any prompt mismatches

The output structure exactly follows the existing JSON files in completion_final/.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def validate_file_structure(data: Dict[str, Any], file_path: str) -> None:
    """Validate that the JSON file has the expected structure."""
    if 'meta' not in data:
        raise ValueError(f"File {file_path} missing 'meta' field")
    if 'items' not in data:
        raise ValueError(f"File {file_path} missing 'items' field")
    
    meta = data['meta']
    required_meta_fields = ['dataset', 'split', 'num_prompts', 'seed', 'model1_path', 'model2_path', 'use_vllm', 'generation_params']
    for field in required_meta_fields:
        if field not in meta:
            raise ValueError(f"File {file_path} missing required meta field: {field}")
    
    items = data['items']
    if not isinstance(items, list):
        raise ValueError(f"File {file_path} 'items' field must be a list")
    
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"File {file_path} item {i} is not a dictionary")
        if 'prompt' not in item:
            raise ValueError(f"File {file_path} item {i} missing 'prompt' field")
        if 'completion1' not in item:
            raise ValueError(f"File {file_path} item {i} missing 'completion1' field")
        if 'completion2' not in item:
            raise ValueError(f"File {file_path} item {i} missing 'completion2' field")


def validate_prompts_match(file1_data: Dict[str, Any], file2_data: Dict[str, Any]) -> None:
    """Validate that both files have identical prompts. Hard stop if not."""
    file1_items = file1_data['items']
    file2_items = file2_data['items']
    
    # Check if both files have the same number of items
    if len(file1_items) != len(file2_items):
        raise ValueError(f"Files have different number of items: {len(file1_items)} vs {len(file2_items)}")
    
    # Check if all prompts match exactly
    for i in range(len(file1_items)):
        prompt1 = file1_items[i]['prompt']
        prompt2 = file2_items[i]['prompt']
        
        if prompt1 != prompt2:
            raise ValueError(f"Prompt mismatch at item {i}:\n"
                           f"File 1: {prompt1[:100]}...\n"
                           f"File 2: {prompt2[:100]}...")
    
    print(f"✓ All {len(file1_items)} prompts match perfectly between the two files")


def create_hybrid_comparison(
    file1_path: str,
    file2_path: str,
    file1_model_choice: str,
    file2_model_choice: str,
    output_path: str
) -> None:
    """
    Create a hybrid comparison JSON file.
    
    Args:
        file1_path: Path to the first JSON file
        file2_path: Path to the second JSON file
        file1_model_choice: Either 'model1' or 'model2' for file1
        file2_model_choice: Either 'model1' or 'model2' for file2
        output_path: Path where to save the new JSON file
    """
    
    print(f"Loading file 1: {file1_path}")
    file1_data = load_json_file(file1_path)
    validate_file_structure(file1_data, file1_path)
    
    print(f"Loading file 2: {file2_path}")
    file2_data = load_json_file(file2_path)
    validate_file_structure(file2_data, file2_path)
    
    print("Validating prompt matches...")
    validate_prompts_match(file1_data, file2_data)
    
    # Extract model paths and completions based on choices
    file1_meta = file1_data['meta']
    file2_meta = file2_data['meta']
    
    if file1_model_choice == 'model1':
        file1_model_path = file1_meta['model1_path']
        file1_completion_key = 'completion1'
    elif file1_model_choice == 'model2':
        file1_model_path = file1_meta['model2_path']
        file1_completion_key = 'completion2'
    else:
        raise ValueError(f"Invalid file1_model_choice: {file1_model_choice}. Must be 'model1' or 'model2'")
    
    if file2_model_choice == 'model1':
        file2_model_path = file2_meta['model1_path']
        file2_completion_key = 'completion1'
    elif file2_model_choice == 'model2':
        file2_model_path = file2_meta['model2_path']
        file2_completion_key = 'completion2'
    else:
        raise ValueError(f"Invalid file2_model_choice: {file2_model_choice}. Must be 'model1' or 'model2'")
    
    print(f"File 1: Using {file1_model_choice} -> {file1_model_path}")
    print(f"File 2: Using {file2_model_choice} -> {file2_model_path}")
    
    # Create new items with selected completions
    new_items = []
    file1_items = file1_data['items']
    file2_items = file2_data['items']
    
    for i in range(len(file1_items)):
        new_item = {
            'prompt': file1_items[i]['prompt'],  # Same from both files
            'completion1': file1_items[i][file1_completion_key],
            'completion2': file2_items[i][file2_completion_key]
        }
        new_items.append(new_item)
    
    # Create new meta information
    new_meta = {
        'dataset': file1_meta['dataset'],
        'split': file1_meta['split'],
        'num_prompts': file1_meta['num_prompts'],
        'seed': file1_meta['seed'],
        'model1_path': file1_model_path,
        'model2_path': file2_model_path,
        'use_vllm': file1_meta['use_vllm'],
        'generation_params': file1_meta['generation_params']
    }
    
    # Create the new JSON structure
    new_data = {
        'meta': new_meta,
        'items': new_items
    }
    
    # Save the new JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully created hybrid comparison file: {output_path}")
    print(f"  - {len(new_items)} items processed")
    print(f"  - Model 1: {file1_model_path}")
    print(f"  - Model 2: {file2_model_path}")


def list_available_files(completion_final_dir: str) -> List[str]:
    """List all JSON files in the completion_final directory."""
    completion_final_path = Path(completion_final_dir)
    if not completion_final_path.exists():
        raise FileNotFoundError(f"Directory not found: {completion_final_dir}")
    
    json_files = list(completion_final_path.glob("*.json"))
    return [str(f) for f in json_files]


def main():
    """Main function to run the script interactively."""
    completion_final_dir = "completion_final"
    
    print("=== Hybrid Comparison JSON Creator ===")
    print()
    
    # List available files
    try:
        available_files = list_available_files(completion_final_dir)
        print(f"Found {len(available_files)} JSON files in {completion_final_dir}:")
        for i, file_path in enumerate(available_files):
            filename = os.path.basename(file_path)
            print(f"  {i+1:2d}. {filename}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get user input
    try:
        # File selection
        file1_idx = int(input("Enter the number for the first file: ")) - 1
        if file1_idx < 0 or file1_idx >= len(available_files):
            raise ValueError("Invalid file number")
        
        file2_idx = int(input("Enter the number for the second file: ")) - 1
        if file2_idx < 0 or file2_idx >= len(available_files):
            raise ValueError("Invalid file number")
        
        if file1_idx == file2_idx:
            raise ValueError("Cannot select the same file twice")
        
        file1_path = available_files[file1_idx]
        file2_path = available_files[file2_idx]
        
        print(f"\nSelected files:")
        print(f"  File 1: {os.path.basename(file1_path)}")
        print(f"  File 2: {os.path.basename(file2_path)}")
        
        # Model choice for each file
        print(f"\nFor file 1 ({os.path.basename(file1_path)}):")
        file1_choice = input("Choose model (1 for model1, 2 for model2): ").strip()
        if file1_choice == '1':
            file1_model_choice = 'model1'
        elif file1_choice == '2':
            file1_model_choice = 'model2'
        else:
            raise ValueError("Invalid choice. Must be 1 or 2")
        
        print(f"\nFor file 2 ({os.path.basename(file2_path)}):")
        file2_choice = input("Choose model (1 for model1, 2 for model2): ").strip()
        if file2_choice == '1':
            file2_model_choice = 'model1'
        elif file2_choice == '2':
            file2_model_choice = 'model2'
        else:
            raise ValueError("Invalid choice. Must be 1 or 2")
        
        # Generate output filename
        file1_name = os.path.basename(file1_path).replace('.json', '')
        file2_name = os.path.basename(file2_path).replace('.json', '')
        
        # Extract model names from the chosen paths
        file1_data = load_json_file(file1_path)
        file2_data = load_json_file(file2_path)
        
        if file1_model_choice == 'model1':
            model1_name = file1_data['meta']['model1_path'].split('/')[-1]
        else:
            model1_name = file1_data['meta']['model2_path'].split('/')[-1]
        
        if file2_model_choice == 'model1':
            model2_name = file2_data['meta']['model1_path'].split('/')[-1]
        else:
            model2_name = file2_data['meta']['model2_path'].split('/')[-1]
        
        # Create output filename following the existing pattern
        num_prompts = file1_data['meta']['num_prompts']
        seed = file1_data['meta']['seed']
        output_filename = f"{model1_name}_vs_{model2_name}_{num_prompts}prompts_seed{seed}_omit_thinking_new_instruct.json"
        output_path = os.path.join(completion_final_dir, output_filename)
        
        print(f"\nOutput file will be: {output_filename}")
        
        # Confirm before proceeding
        confirm = input("\nProceed with creating the hybrid comparison? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        
        # Create the hybrid comparison
        create_hybrid_comparison(
            file1_path, file2_path,
            file1_model_choice, file2_model_choice,
            output_path
        )
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()