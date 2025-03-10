import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Now run the training script
if __name__ == "__main__":
    from scripts.train_models import train_models
    import asyncio
    asyncio.run(train_models())
