from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



root_dir = Path("archive/chest_xray")

print(f"Dataset: {root_dir}")

train_dir = root_dir / "train"
test_dir = root_dir / "test"
val_dir = root_dir / "val"

for folder in [train_dir, test_dir, val_dir]:
    if folder.exists():
        normal_count = len(list((folder / "NORMAL").glob("*.jpeg")))
        pneumonia_count = len(list((folder / "PNEUMONIA").glob("*.jpeg")))
        print(f"   -> {folder.name.upper()}: {normal_count} Normal, {pneumonia_count} Pneumonia")
    else:
        print(f"   -> {folder.name.upper()}: ❌ Not found (Check path!)")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

