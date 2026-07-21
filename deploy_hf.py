"""
Automatic Hugging Face 100% Free Model Hub Deployment Script for Daksh AI
Upload Fine-Tuned Model Weights to Hugging Face Model Hub
"""

import os
import sys
import shutil
from huggingface_hub import HfApi, create_repo

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def deploy(hf_token: str, model_name: str = "daksh-coding-assistant"):
    print("=" * 65)
    print("🚀 Uploading Fine-Tuned Model to Hugging Face Model Hub (100% FREE)...")
    print("=" * 65)

    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info['name']

    repo_id = f"{username}/{model_name}"
    print(f"[+] Hugging Face Username: {username}")
    print(f"[+] Target Model Repository: {repo_id}")

    # Step 1: Create Model Repo on HF Model Hub (100% Free for everyone)
    try:
        print("[+] Creating Hugging Face Model Repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            exist_ok=True
        )
        print("✅ Model repository created successfully!")
    except Exception as e:
        print(f"[!] Repo creation info: {e}")

    model_dir = os.path.join(os.path.dirname(__file__), "models", "coding-assistant")
    if not os.path.exists(model_dir):
        print(f"[!] Error: Model directory '{model_dir}' not found.")
        return

    # Step 2: Upload Fine-Tuned LoRA model weights
    print(f"[+] Uploading Fine-Tuned LoRA weights to '{repo_id}'...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )

    model_url = f"https://huggingface.co/{repo_id}"
    print("\n" + "=" * 65)
    print("🎉 MODEL UPLOAD SUCCESSFUL!")
    print("🌍 Your Permanent Hugging Face Model Hub URL:")
    print(f"👉 {model_url}")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
        deploy(token)
    else:
        print("Usage: python deploy_hf.py <YOUR_HF_TOKEN>")
