#!/usr/bin/env python3
"""
Script to upload existing HF model directories to the hub.
Supports both individual repos and single repo with subdirectories.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_main_repo(repo_name, private=False):
    """
    Create the main repository on HF Hub
    
    Args:
        repo_name: Name for the hub repository (e.g., "username/repo-name")
        private: Whether to make the hub repository private
    """
    print(f"Creating main repository: {repo_name}")
    
    try:
        from huggingface_hub import create_repo
        # Create the repo
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"✓ Main repository created: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False


def upload_model_to_subdir(model_dir, repo_name, model_subdir, private=False):
    """
    Upload a model to a subdirectory within the main repo
    
    Args:
        model_dir: Path to the existing HF model directory
        repo_name: Name for the hub repository (e.g., "username/repo-name")
        model_subdir: Subdirectory name for this specific model
        private: Whether the repo is private
    """
    print(f"Uploading model to subdirectory: {model_subdir}")
    
    try:
        from huggingface_hub import upload_folder
        # Upload the entire model directory to the subdirectory
        upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            path_in_repo=model_subdir,
            commit_message=f"Add {model_subdir} model"
        )
        
        print(f"✓ Model uploaded to: {repo_name}/{model_subdir}")
        return True
    except Exception as e:
        print(f"Error uploading model: {e}")
        return False


def upload_hf_model_individual(model_dir, repo_name, private=False):
    """
    Upload an existing HF model directory to its own repo
    
    Args:
        model_dir: Path to the existing HF model directory
        repo_name: Name for the hub repository (e.g., "username/model-name")
        private: Whether to make the hub repository private
    """
    print(f"Uploading existing HF model from: {model_dir}")
    
    # Load the model and tokenizer from the existing directory
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Push to hub
    print(f"Pushing model to hub: {repo_name}")
    model.push_to_hub(repo_name, private=private)
    tokenizer.push_to_hub(repo_name, private=private)
    
    print(f"✓ Model successfully uploaded to hub: {repo_name}")
    return repo_name


def main():
    parser = argparse.ArgumentParser(description="Upload existing HF model directory to hub")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to existing HF model directory")
    parser.add_argument("--repo-name", type=str, required=True,
                       help="Name for the hub repository (e.g., 'username/repo-name')")
    parser.add_argument("--model-subdir", type=str,
                       help="Subdirectory name for this model within the repo (for single repo mode)")
    parser.add_argument("--private", action="store_true",
                       help="Make the hub repository private")
    parser.add_argument("--create-repo", action="store_true",
                       help="Create the main repository (only needed once for single repo mode)")
    parser.add_argument("--single-repo", action="store_true",
                       help="Upload to subdirectory within single repo instead of creating individual repo")
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    # Check if HF token is set
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Warning: HF_TOKEN not set. Make sure you're logged in with 'huggingface-cli login'")
    
    try:
        if args.single_repo:
            # Single repo mode with subdirectories
            if not args.model_subdir:
                print("Error: --model-subdir is required when using --single-repo")
                return
                
            # Create main repo if requested
            if args.create_repo:
                if not create_main_repo(args.repo_name, args.private):
                    return
            
            # Upload the model to subdirectory
            if upload_model_to_subdir(args.model_dir, args.repo_name, args.model_subdir, args.private):
                print("\n" + "="*60)
                print("UPLOAD COMPLETE!")
                print("="*60)
                print(f"Model uploaded to: https://huggingface.co/{args.repo_name}/{args.model_subdir}")
                print(f"Main repository: https://huggingface.co/{args.repo_name}")
                print(f"Model subdirectory: {args.model_subdir}")
                print(f"You can use it in model_comparison.py with: --model1 '{args.repo_name}/{args.model_subdir}'")
                print("="*60)
        else:
            # Individual repo mode
            hub_name = upload_hf_model_individual(
                args.model_dir, 
                args.repo_name, 
                args.private
            )
            
            print("\n" + "="*60)
            print("UPLOAD COMPLETE!")
            print("="*60)
            print(f"Your model is now available at: https://huggingface.co/{hub_name}")
            print(f"You can use it in model_comparison.py with: --model1 '{hub_name}'")
            print("="*60)
        
    except Exception as e:
        print(f"Error during upload: {e}")
        print("Make sure you're logged in to HF Hub and have write access to the repository")


if __name__ == "__main__":
    main() 