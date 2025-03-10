import os
from config import config

def cleanup_models():
    model_dir = config.MODEL_DIR
    count = 0
    for file in model_dir.glob("*.keras"):
        print(f"Deleting {file}")
        file.unlink()
        count += 1
    print(f"Deleted {count} model file(s).")

if __name__ == "__main__":
    cleanup_models()
