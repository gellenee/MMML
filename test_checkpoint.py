# Force the dimensions to match the 1024/4096 found in the checkpoint
class LargeConfig:
    def __init__(self):
        self.batch_size = 8
        self.dataset_name = 'mosi'
        self.model = 'cme'
        self.cme_version = 'v1' # Check if this should be v3 based on your previous error
        self.dropout = 0.3
        self.num_hidden_layers = 5 # As confirmed by "CME_layers.4"
        self.tasks = 'MTA'
        self.train_mode = 'regression'

# You MUST pass these specific numbers to BertConfig now
from utils.cross_attn_encoder import BertConfig

# Create a config that matches the checkpoint's "Brain"
custom_bert_config = BertConfig(
    hidden_size=1024, 
    intermediate_size=4096, 
    num_hidden_layers=5
)

# When building the model, ensure it uses these specs
model = rob_d2v_cme(LargeConfig()) 

# CRITICAL: Overwrite the automatically created CME layers with Large ones
from utils.cross_attn_encoder import CMELayer
import torch.nn as nn

model.CME_layers = nn.ModuleList(
    [CMELayer(custom_bert_config) for _ in range(5)]
)