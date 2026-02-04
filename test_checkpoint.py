import torch
from utils.en_train import EnConfig, EnRun
from utils.ch_model import rob_d2v_cc, rob_d2v_cme  # These are the models from your model file
from utils.data_loader import data_loader
from utils.metricsTop import MetricsTop

# 1. Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_checkpoint(checkpoint_path):
    # 2. Use the exact default values from your argparse
    config = EnConfig(
        seed=1,
        batch_size=8,
        learning_rate=5e-6,
        model='cme',
        cme_version='v1',
        dataset_name='mosi',
        num_hidden_layers=5,
        tasks='MTA',
        context=True,
        text_context_len=2,
        audio_context_len=1
    )

    # 3. Build the Model (The Skeleton)
    print(f"Building {config.model} model architecture...")
    if config.model == 'cc':
        model = rob_d2v_cc(config).to(device)
    else:
        model = rob_d2v_cme(config).to(device)

    # 4. Load the Weights (The Brain)
    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Load the Test Data
    print(f"Loading {config.dataset_name} test dataset...")
    # Based on your loader, we ignore train and val
    _, test_loader, _ = data_loader(config.batch_size, config.dataset_name)

    # 6. Evaluation
    metrics = MetricsTop('regression').getMetics(config.dataset_name)
    y_pred = {'M': [], 'T': [], 'A': []}
    y_true = {'M': [], 'T': [], 'A': []}

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            text_inputs = batch["text_tokens"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"]

            outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

            for m in config.tasks:
                y_pred[m].append(outputs[m].cpu())
                y_true[m].append(targets[m].cpu())

    # 7. Print Results
    print("\n--- TEST RESULTS ---")
    for m in config.tasks:
        pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
        res = metrics(pred, true)
        print(f"Task {m}: {res}")

if __name__ == "__main__":
    # Point to your backup file
    PATH = "backups/RH_acc_mosi_1_0.8734.pth"
    test_checkpoint(PATH)