import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.context_model import rob_d2v_cc_context, rob_d2v_cme_context
from utils.en_model import rob_d2v_cc, rob_d2v_cme
from utils.en_model_tav import rob_d2v_videomae_cme
import random
import numpy as np
from utils.data_loader import data_loader
from itertools import chain
import os
# global variable
gpu_id = int(os.environ.get('CUDA_DEVICE_ID', '0'))
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


class EnConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M':1,
                    'T':1,
                    'A':1,
                    'V':1,
                },
                 model_save_path = 'checkpoint/',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 8,
                 seed = 0,
                 dropout=0.3,
                 model='cc',
                 batch_size = 16,
                 multi_task = True,
                 model_size = 'small',
                 cme_version = 'v1',
                 num_hidden_layers = 1,
                 tasks = 'M',   # 'M' or 'MTA',
                 context = True,
                 text_context_len = 2,
                 audio_context_len = 1,
                 modalities = 'TV',
                 
                ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.model_size = model_size
        self.cme_version = cme_version
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.modalities = modalities

        
        
class EnTrainer():
    def __init__(self, config):
 
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader):    
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0
        dataset_len = len(data_loader.dataset)

        use_video = "V" in self.config.modalities
        use_audio = "A" in self.config.modalities
        # Loop over all batches.         
        for batch in tqdm(data_loader):        
                        
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            optimizer.zero_grad()

            if use_video:
                video_pixel_values = batch["video_pixel_values"].to(device)

                video_mask = batch["video_mask"].to(device) if "video_mask" in batch else None
                
                audio_inputs = batch["audio_inputs"].to(device) if use_audio and "audio_inputs" in batch else None
                audio_mask = batch["audio_masks"].to(device) if use_audio and "audio_masks" in batch else None
                outputs = model(
                text_inputs,
                text_mask,
                audio_inputs=audio_inputs,
                audio_mask=audio_mask,
                video_pixel_values=video_pixel_values,
                video_mask=video_mask,
                )
            else:
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                if self.config.context:
                    text_context_inputs = batch["text_context_tokens"].to(device)
                    text_context_mask = batch["text_context_masks"].to(device)
                    audio_context_inputs = batch["audio_context_inputs"].to(device)
                    audio_context_mask = batch["audio_context_masks"].to(device)
                    outputs = model(
                        text_inputs,
                        text_mask,
                        text_context_inputs,
                        text_context_mask,
                        audio_inputs,
                        audio_mask,
                        audio_context_inputs,
                        audio_context_mask,
                    )
                else:
                    outputs = model(
                        text_inputs,
                        text_mask,
                        audio_inputs,
                        audio_mask,
                    )
            
            # Compute the training loss.
            if self.config.multi_task:
                active_tasks = [t for t in self.tasks if (t in outputs and t in self.config.loss_weights)]
                loss = 0.0         
                for m in active_tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
    #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss = self.criterion(outputs['M'], targets)        
                total_loss += loss.item()*text_inputs.size(0)
        
            loss.backward()                   
            optimizer.step()    
        if not hasattr(self, "_debug_printed_train_batch"):
            self._debug_printed_train_batch = False

        if not self._debug_printed_train_batch:
            print("\n[DEBUG] ===== First train batch =====")
            print(f"[DEBUG] batch_keys={list(batch.keys())}")
            print(f"[DEBUG] has_video_pixel_values={'video_pixel_values' in batch}")
            if "video_pixel_values" in batch:
                print(f"[DEBUG] video_pixel_values_shape={tuple(batch['video_pixel_values'].shape)}")
            print(f"[DEBUG] text_tokens_shape={tuple(batch['text_tokens'].shape)}")
            if "audio_inputs" in batch:
                print(f"[DEBUG] audio_inputs_shape={tuple(batch['audio_inputs'].shape)}")
            print("[DEBUG] =============================\n")
            self._debug_printed_train_batch = True            
                    
            total_loss = round(total_loss / len(data_loader.dataset), 4)
#         print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.

        use_video = "V" in self.config.modalities
        use_audio = "A" in self.config.modalities


        dataset_len = len(data_loader.dataset)
        total_loss = 0.0

        active_tasks = None
        y_pred = None
        y_true = None
        val_loss = None

        if self.config.multi_task:
            pass
        else:
            y_pred = []
            y_true = []

        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1)
            
                if use_video:
                    video_pixel_values = batch["video_pixel_values"].to(device)
                    video_mask = batch["video_mask"].to(device) if "video_mask" in batch else None

                    audio_inputs = batch["audio_inputs"].to(device) if use_audio and "audio_inputs" in batch else None
                    audio_mask = batch["audio_masks"].to(device) if use_audio and "audio_masks" in batch else None
                    outputs = model(
                        text_inputs,
                        text_mask,
                        audio_inputs=audio_inputs,
                        audio_mask=audio_mask,
                        video_pixel_values=video_pixel_values,
                        video_mask=video_mask,
                    )
                else:
                    audio_inputs = batch["audio_inputs"].to(device)
                    audio_mask = batch["audio_masks"].to(device)
                    if self.config.context:
                        text_context_inputs = batch["text_context_tokens"].to(device)
                        text_context_mask = batch["text_context_masks"].to(device)
                        audio_context_inputs = batch["audio_context_inputs"].to(device)
                        audio_context_mask = batch["audio_context_masks"].to(device)
                        outputs = model(
                            text_inputs,
                            text_mask,
                            text_context_inputs,
                            text_context_mask,
                            audio_inputs,
                            audio_mask,
                            audio_context_inputs,
                            audio_context_mask,
                        )
                    else:
                        outputs = model(
                            text_inputs,
                            text_mask,
                            audio_inputs,
                            audio_mask,
                        )

                # Compute loss.
                if self.config.multi_task:
                    if active_tasks is None:
                        active_tasks = [t for t in self.tasks if (t in outputs and t in self.config.loss_weights)]
                        y_pred = {t: [] for t in active_tasks}
                        y_true = {t: [] for t in active_tasks}
                        val_loss = {t: 0.0 for t in active_tasks}

                    loss = 0.0         
                    for m in active_tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                    total_loss += loss.item()*text_inputs.size(0)
                    
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())
        # aggregate results
        if self.config.multi_task:

            for m in active_tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(val_loss[m] /dataset_len, 4)
            loss_str = " ".join([f"{m}_loss: {val_loss[m]:.4f}" for m in active_tasks])
            print(mode + " >> loss: ", total_loss, "   " + loss_str)

            eval_results = {}
            for m in active_tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            primary_task = self.tasks[0] if self.tasks[0] in eval_results else "M"
            eval_results = eval_results[primary_task] if primary_task in eval_results else eval_results[active_tasks[0]]
            eval_results['Loss'] = total_loss 
            return eval_results
        else:
            total_loss = round(total_loss / dataset_len, 4)
            print(mode + " >> loss: ", total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            eval_results['Loss'] = total_loss
            return eval_results


def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name, modalities=config.modalities,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len)

    use_video = "V" in config.modalities
    if use_video:
        model = rob_d2v_videomae_cme(config).to(device)
    else:
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

    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0

    print("\n[DEBUG] ===== EnRun model/data sanity =====")
    print(f"[DEBUG] dataset_name={config.dataset_name}")
    print(f"[DEBUG] modalities={config.modalities}")
    print(f"[DEBUG] use_video={'V' in config.modalities}")
    print(f"[DEBUG] model_class={model.__class__.__name__}")

    sd_keys = list(model.state_dict().keys())
    print(f"[DEBUG] state_dict_num_keys={len(sd_keys)}")
    print(f"[DEBUG] has_prefix_videomae={any(k.startswith('videomae.') for k in sd_keys)}")
    print(f"[DEBUG] has_prefix_roberta={any(k.startswith('roberta.') for k in sd_keys)}")
    print(f"[DEBUG] has_prefix_roberta_model={any(k.startswith('roberta_model.') for k in sd_keys)}")
    print(f"[DEBUG] first_100_keys={sd_keys[:100]}")
    print("[DEBUG] ====================================\n")

    print("\n[DEBUG] ===== Module tree / VideoMAE check =====")
    module_names = [name for name, _ in model.named_modules()]
    param_names = [name for name, _ in model.named_parameters()]
    state_keys = list(model.state_dict().keys())

    # 1) Find modules that look video-related
    video_module_hits = [
        n for n in module_names
        if ("video" in n.lower()) or ("videomae" in n.lower())
    ]
    print(f"[DEBUG] video-related modules ({len(video_module_hits)}):")
    for n in video_module_hits[:100]:
        print(f"  - {n}")

    # 2) Find params/state keys under those modules
    video_param_hits = [
        n for n in param_names
        if n.startswith("videomae.") or ("video" in n.lower())
    ]
    video_state_hits = [
        k for k in state_keys
        if k.startswith("videomae.") or ("video" in k.lower())
    ]

    print(f"[DEBUG] video-related named_parameters ({len(video_param_hits)}):")
    for n in video_param_hits[:120]:
        print(f"  - {n}")

    print(f"[DEBUG] video-related state_dict keys ({len(video_state_hits)}):")
    for k in video_state_hits[:120]:
        print(f"  - {k}")

    # 3) Strong assertions as booleans
    has_videomae_module = any(n == "videomae" or n.startswith("videomae.") for n in module_names)
    has_videomae_params = any(n.startswith("videomae.") for n in param_names)
    has_videomae_state = any(k.startswith("videomae.") for k in state_keys)

    print(f"[DEBUG] has_videomae_module={has_videomae_module}")
    print(f"[DEBUG] has_videomae_params={has_videomae_params}")
    print(f"[DEBUG] has_videomae_state={has_videomae_state}")
    print("[DEBUG] =========================================\n")

    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")

        # Save the best LOSS model (Overwrites previous best)
        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            save_path_loss = config.model_save_path + f'best_loss_{config.dataset_name}_seed{config.seed}_loss{eval_results["Loss"]:.4f}.pth'
            torch.save(model.state_dict(), save_path_loss)
            best_epoch = epoch
            
        # Save the best ACCURACY model (Overwrites previous best)
        if eval_results['Has0_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            save_path_acc = config.model_save_path + f'best_loss_{config.dataset_name}_seed{config.seed}_loss{eval_results["Loss"]:.4f}.pth'
            torch.save(model.state_dict(), save_path_acc)
            
        if epoch - best_epoch >= config.early_stop:
            break

    # --- FINAL TESTING ---
    
    # Load and test the highest accuracy version
    print("\nLoading Best Accuracy Model...")
    model.load_state_dict(torch.load(save_path_acc))        
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_acc))

    # Load and test the lowest loss version
    print("\nLoading Best Loss Model...")
    model.load_state_dict(torch.load(save_path_loss))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_loss))