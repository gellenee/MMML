import torch
from utils.ch_model import rob_hub_cc, rob_hub_cme
from utils.data_loader import data_loader

# 1. Setup Configuration (Match the settings used during training)
config = ChConfig(
    dataset_name = 'mosi', 
    model = 'cme',        # or 'cc'
    tasks = 'MTA'         # match your training tasks
)

# 2. Initialize the Model Architecture
if config.model == 'cc':
    model = rob_hub_cc(config).to(device)
else:
    model = rob_hub_cme(config).to(device)

# 3. Load the .pth file
# Replace this path with the one you moved to the backup folder
checkpoint_path = 'backups/RH_acc_mosi_1_0.8734.pth'

print(f"Loading weights from {checkpoint_path}...")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 4. Prepare Data & Trainer
_, test_loader, _ = data_loader(config.batch_size, config.dataset_name)
trainer = ChTrainer(config)

# 5. Run the Test
print("Starting Evaluation...")
test_results = trainer.do_test(model, test_loader, "FINAL TEST")