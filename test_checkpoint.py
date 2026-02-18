import torch
import random
import numpy as np
from utils.ch_train import ChConfig, ChTrainer, dict_to_str  # Changed from en_train
from utils.ch_model import rob_hub_cc, rob_hub_cme  # Changed from en_model
from utils.data_loader import data_loader
from utils.metricsTop import MetricsTop

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_checkpoint(checkpoint_path, config=None, dataset='mosei'):
    """
    Load a checkpoint and test it on dataset evaluation.
    
    Args:
        checkpoint_path: Path to your .pth checkpoint file
        config: Optional Config object. If None, will create a default config.
        dataset: 'sims', 'mosi', or 'mosei'
    """
    # Set seed
    set_seed(1)
    
    # Use Chinese models/config for SIMS, English for MOSI/MOSEI
    if dataset == 'sims':
        from utils.ch_train import ChConfig, ChTrainer
        from utils.ch_model import rob_hub_cc, rob_hub_cme
        
        if config is None:
            config = ChConfig(
                train_mode='regression',  # or 'classification'
                dataset_name='sims',
                model='cme',
                cme_version='v1',
                num_hidden_layers=5,
                batch_size=8,
                tasks='MTA',  # SIMS uses multi-task
                multi_task=True,
                dropout=0.3
            )
        
        # Load data
        print("Loading test data...")
        train_loader, test_loader, val_loader = data_loader(
            config.batch_size, 
            config.dataset_name
        )
        
        # Initialize Chinese model
        print("Initializing model...")
        if config.model == 'cc':
            model = rob_hub_cc(config).to(device)
        elif config.model == 'cme':
            model = rob_hub_cme(config).to(device)
        for param in model.hubert_model.feature_extractor.parameters():
            param.requires_grad = False
        
        # Initialize trainer
        trainer = ChTrainer(config)
        
    else:  # MOSI or MOSEI
        from utils.en_train import EnConfig, EnTrainer
        from utils.context_model import rob_d2v_cc_context, rob_d2v_cme_context
        from utils.en_model import rob_d2v_cc, rob_d2v_cme
        
        if config is None:
            config = EnConfig(
                train_mode='regression',
                dataset_name=dataset,
                model='cme',
                cme_version='v1',
                num_hidden_layers=5,
                batch_size=8,
                context=True,
                text_context_len=2,
                audio_context_len=1,
                tasks='M',
                multi_task=False,
                dropout=0.3
            )
        
        # Load data
        print("Loading test data...")
        train_loader, test_loader, val_loader = data_loader(
            config.batch_size, 
            config.dataset_name,
            text_context_length=config.text_context_len,
            audio_context_length=config.audio_context_len
        )
        
        # Initialize English model
        print("Initializing model...")
        if config.context:
            if config.model == 'cc':
                model = rob_d2v_cc_context(config).to(device)
            elif config.model == 'cme':
                model = rob_d2v_cme_context(config).to(device)
            for param in model.data2vec_model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            if config.model == 'cc':
                model = rob_d2v_cc(config).to(device)
            elif config.model == 'cme':
                model = rob_d2v_cme(config).to(device)
            for param in model.data2vec_model.feature_extractor.parameters():
                param.requires_grad = False
        
        # Initialize trainer
        trainer = EnTrainer(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Test on test set
    print("\n" + "="*50)
    print("Testing on TEST set...")
    print("="*50)
    test_results = trainer.do_test(model, test_loader, "TEST")
    print(f'\nTEST Results: {dict_to_str(test_results)}')
    
    # Also test on validation set if needed
    print("\n" + "="*50)
    print("Testing on VALIDATION set...")
    print("="*50)
    val_results = trainer.do_test(model, val_loader, "VAL")
    print(f'\nVAL Results: {dict_to_str(val_results)}')
    
    return test_results, val_results

# Example usage:
if __name__ == "__main__":
    # For SIMS dataset
    checkpoint_path = "checkpoint/acc_seed111.pth"  # Update this path
    
    from utils.ch_train import ChConfig
    from utils.en_train import EnConfig
    
    ch_config = ChConfig(
        train_mode='regression',  # or 'classification'
        dataset_name='sims',
        model='cme',
        cme_version='v1',
        num_hidden_layers=5,  # Make sure this matches your training config
        batch_size=8,
        tasks='MTA',  # SIMS uses multi-task
        dropout=0.3
    )

    en_config = EnConfig(
                train_mode='regression',
                dataset_name="mosei",
                model='cme',
                cme_version='v1',
                num_hidden_layers=5,
                batch_size=8,
                context = True,
                text_context_len=2,
                audio_context_len=1,
                tasks='M',
                multi_task=False,
                dropout=0.3
            )
    # Run test
    test_results, val_results = test_checkpoint(checkpoint_path, config=ch_config, dataset='sims')
    
    # Print metrics summary
    print("\n" + "="*50)
    print("MOSEI Metrics Summary:")
    print("="*50)
    if 'Mult_acc_2' in test_results:
        print(f"Mult_acc_2: {test_results.get('Mult_acc_2', 'N/A')}")
        print(f"Mult_acc_3: {test_results.get('Mult_acc_3', 'N/A')}")
        print(f"Mult_acc_5: {test_results.get('Mult_acc_5', 'N/A')}")
        print(f"F1_score: {test_results.get('F1_score', 'N/A')}")
        print(f"MAE: {test_results.get('MAE', 'N/A')}")
        print(f"Corr: {test_results.get('Corr', 'N/A')}")
    else:
        for key, value in test_results.items():
            print(f"{key}: {value}")