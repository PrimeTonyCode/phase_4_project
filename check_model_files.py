import os

def check_model_files():
    """
    Check if all necessary model files exist in the root and bert_deployment directories
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bert_dir = os.path.join(base_dir, "bert_deployment")
    
    # Files that should be present for BERT model to work
    required_files = [
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json", 
        "vocab.txt",
        "config.json"
    ]
    
    # Check files in bert_deployment directory
    print("Checking bert_deployment directory...")
    bert_missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(bert_dir, file)):
            bert_missing.append(file)
    
    if bert_missing:
        print(f"Missing files in bert_deployment: {bert_missing}")
    else:
        print("All required model files present in bert_deployment directory.")
    
    # Check if we need to copy files to the root directory
    print("\nEnsuring model files are available in the root directory...")
    for file in required_files:
        source_path = os.path.join(bert_dir, file)
        dest_path = os.path.join(base_dir, file)
        
        if os.path.exists(source_path) and not os.path.exists(dest_path):
            print(f"Need to copy: {file} to root directory")
        elif not os.path.exists(dest_path):
            print(f"Warning: {file} missing in both directories")
        else:
            print(f"File exists in root directory: {file}")

if __name__ == "__main__":
    check_model_files()
