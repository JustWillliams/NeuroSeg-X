import os
import torch
from data_loader import setup_colab_data, NeuroSegDataset, get_transforms
from neuroseg_x import NeuroSegX
from baselines import UNet, TransUNet, SwinUNet, BiTrUNet
from trainer import MCDOTrainer
from torch.utils.data import DataLoader, random_split

# --- configuration ---
config = {
    'batch_size': 4,
    'lr': 1e-4,
    'num_epochs': 50,
    'img_size': (512, 512),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'drive_path': '/content/drive/MyDrive/Colab_Notebooks_Data',
    'log_dir': '/content/runs'
}

def main():
    print(f"Starting NeuroSeg-X on {config['device']}")
    
    # 1. Setup Data
    data_path = setup_colab_data(config['drive_path'])
    
    # Placeholder for finding image and mask paths in extracted directories
    # In a real scenario, this would involve globbing the extracted paths
    image_paths = [] # list(Path(data_path).rglob('*.nii.gz')) -> convert to png/npy/etc
    mask_paths = []
    labels = {'detection': [], 'grading': []}
    
    # 2. Get Transforms
    train_tf, val_tf = get_transforms(config['img_size'])
    
    # 3. Create Dataset and Split
    dataset = NeuroSegDataset(image_paths, mask_paths, labels, transforms=train_tf)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    
    # 4. Initialize NeuroSeg-X
    model = NeuroSegX(in_channels=3, seg_classes=4).to(config['device'])
    
    # 5. Initialize Trainer
    trainer = MCDOTrainer(model, train_loader, val_loader, config, config['device'])
    
    # 6. Start Training Loop
    for epoch in range(config['num_epochs']):
        avg_train_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Loss: {avg_train_loss:.4f}")
        
    print("Training finished!")

if __name__ == "__main__":
    main()
