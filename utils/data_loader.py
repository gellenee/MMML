import torch
import os
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
from utils.video_data import collate_vce_text_video, Dataset_vce_custom_text_video

import pandas as pd
import numpy as np
import string


class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):       
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        self.targets_M = df['label']
        self.targets_T = df['label_T']
        self.targets_A = df['label_A']
        
        # store texts
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # store audio
        self.audio_file_paths = []

        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
      
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   
        
        
    def __getitem__(self, index):
       # extract text features
        text = str(self.texts[index])         
        tokenized_text = self.tokenizer(
            text,            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )               
                
        # extract audio features    
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
            
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "target": {
                    "M": float(self.targets_M.iloc[index]),
                    "T": float(self.targets_T.iloc[index]),
                    "A": float(self.targets_A.iloc[index])
                }
                }
    
    def __len__(self):
        return len(self.targets_M)



class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length
    
    def __init__(self, csv_path, audio_directory, mode, text_context_length=2, audio_context_length=1):
        df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav', '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            df = df[~((df['video_id']==video_id) & (df['clip_id']==int(clip_id)))]

        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()
        
        # store labels
        self.targets_M = df['label']
        
        # store texts
        df['text'] = df['text'].str[0]+df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store audio
        self.audio_file_paths = []
        ## loop through the csv entries
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        # store context
        self.video_id = df['video_id']
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length
        
    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])             

        # load text context
        text_context = ''
        for i in range(1, self.text_context_length+1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context
        
        # tokenize text
        tokenized_text = self.tokenizer(
                text,            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )  
        
        # tokenize text context
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,            
            max_length = 96,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )

        # load audio
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # load audio context
        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length+1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context,_ = torchaudio.load(self.audio_file_paths[index - i])
                contextData = torch.mean(context, dim=0, keepdim=False)
                audio_context = torch.cat((contextData, audio_context), 0)

        # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        # extract audio context features
        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000)
            audio_context_masks = torch.zeros(96000)
        else:
            features = self.feature_extractor(audio_context, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            audio_context_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_context_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
                "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
                "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "audio_context_inputs": audio_context_features,
                "audio_context_masks": audio_context_masks,
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }
    
    def __len__(self):
        return len(self.targets_M)
    
class Dataset_vce_custom(torch.utils.data.Dataset):
    """
    Dataset for VCE_CUSTOM. CSV columns: transcription, video_id, audio_file, mode, text, label, pred_label.
    Audio path = os.path.join(audio_directory, audio_file).
    """
    def __init__(self, csv_path, audio_directory, mode, text_context_length=2, audio_context_length=1):
        df = pd.read_csv(csv_path)
        df = df[df['mode'] == mode].reset_index(drop=True)
        df = df.sort_values(by=['video_id']).reset_index(drop=True)

        self.targets_M = df['label'].astype(np.float32)
        self.texts = df['text'].apply(lambda x: str(x)[0] + str(x)[1:].lower() if len(str(x)) > 1 else str(x))
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self.audio_file_paths = []
        for i in range(len(df)):
            path = os.path.join(audio_directory, str(df['audio_file'].iloc[i]))
            self.audio_file_paths.append(path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        self.video_id = df['video_id'].values
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length

    def __getitem__(self, index):
        text = str(self.texts.iloc[index])

        text_context = ''
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts.iloc[index - i])
                text_context = context + '</s>' + text_context

        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        text_context = text_context[:-4] if text_context.endswith('</s>') else text_context
        tokenized_context = self.tokenizer(
            text_context if text_context.strip() else " ",
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )

        sound, _ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context, _ = torchaudio.load(self.audio_file_paths[index - i])
                contextData = torch.mean(context, dim=0, keepdim=False)
                audio_context = torch.cat((contextData, audio_context), 0)

        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000, return_attention_mask=True, truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000)
            audio_context_masks = torch.zeros(96000)
        else:
            features_ctx = self.feature_extractor(audio_context, sampling_rate=16000, max_length=96000, return_attention_mask=True, truncation=True, padding="max_length")
            audio_context_features = torch.tensor(np.array(features_ctx['input_values']), dtype=torch.float32).squeeze()
            audio_context_masks = torch.tensor(np.array(features_ctx['attention_mask']), dtype=torch.long).squeeze()

        return {
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            "targets": torch.tensor(self.targets_M.iloc[index], dtype=torch.float),
        }

    def __len__(self):
        return len(self.targets_M)
    
class Dataset_vce_custom_tav(torch.utils.data.Dataset):
    """
    VCE_CUSTOM dataset yielding text, audio(+context), and cached video clips.

    Expects CSV `data/VCE_CUSTOM/labels_with_video_cache.csv` columns:
      - video_id
      - audio_file
      - text
      - label
      - mode (train/test/valid)
      - video_cache_file (relative .pt path under data/VCE_CUSTOM/videomae_cache)
    """
    def __init__(self, csv_path, audio_directory, mode,
                 text_context_length=2, audio_context_length=1):

        from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

        df = pd.read_csv(csv_path)
        df = df[df["mode"] == mode].reset_index(drop=True)
        df = df.sort_values(by=["video_id"]).reset_index(drop=True)

        # labels
        self.targets_M = df["label"].astype(np.float32)

        # text
        self.texts = df["text"].apply(
            lambda x: str(x)[0] + str(x)[1:].lower() if len(str(x)) > 1 else str(x)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # audio
        self.audio_file_paths = [
            os.path.join(audio_directory, str(f)) for f in df["audio_file"].tolist()
        ]
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        # video cache
        self.video_cache_files = df["video_cache_file"].astype(str).tolist()
        self.video_cache_root = "data/VCE_CUSTOM/videomae_cache"

        # context bookkeeping
        self.video_id = df["video_id"].values
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length

    def __len__(self):
        return len(self.targets_M)

    def __getitem__(self, index):


        # --- text ---
        text = str(self.texts.iloc[index])

        # text context
        text_context = ""
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context = str(self.texts.iloc[index - i])
            text_context = context + "</s>" + text_context

        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        text_context = text_context[:-4] if text_context.endswith("</s>") else text_context
        tokenized_context = self.tokenizer(
            text_context if text_context.strip() else " ",
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        # --- audio ---
        sound, _ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # audio context
        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context, _ = torchaudio.load(self.audio_file_paths[index - i])
            contextData = torch.mean(context, dim=0, keepdim=False)
            audio_context = torch.cat((contextData, audio_context), 0)

        features = self.feature_extractor(
            soundData,
            sampling_rate=16000,
            max_length=96000,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
        )
        audio_features = torch.tensor(
            np.array(features["input_values"]), dtype=torch.float32
        ).squeeze()
        audio_masks = torch.tensor(
            np.array(features["attention_mask"]), dtype=torch.long
        ).squeeze()

        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000)
            audio_context_masks = torch.zeros(96000)
        else:
            features_ctx = self.feature_extractor(
                audio_context,
                sampling_rate=16000,
                max_length=96000,
                return_attention_mask=True,
                truncation=True,
                padding="max_length",
            )
            audio_context_features = torch.tensor(
                np.array(features_ctx["input_values"]), dtype=torch.float32
            ).squeeze()
            audio_context_masks = torch.tensor(
                np.array(features_ctx["attention_mask"]), dtype=torch.long
            ).squeeze()

        # --- video from cache ---
        cache_rel = self.video_cache_files[index]
        cache_path = os.path.join(self.video_cache_root, cache_rel)
        video_pixel_values = torch.load(cache_path)  # [T,C,H,W]

        return {
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(
                tokenized_context["input_ids"], dtype=torch.long
            ),
            "text_context_masks": torch.tensor(
                tokenized_context["attention_mask"], dtype=torch.long
            ),
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            "video_pixel_values": video_pixel_values,  # [T,C,H,W]
            "targets": torch.tensor(self.targets_M.iloc[index], dtype=torch.float),
        }
def collate_fn_sims(batch):   
    text_tokens = []  
    text_masks = []
    audio_inputs = []  
    audio_masks = []
    
    targets_M = []
    targets_T = []
    targets_A = []
   
    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        #audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])

       # labels
        targets_M.append(batch[i]['target']['M'])
        targets_T.append(batch[i]['target']['T'])
        targets_A.append(batch[i]['target']['A'])        
       
    return {
            # text
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "text_masks": torch.tensor(text_masks, dtype=torch.long),           
            # audio
            "audio_inputs": torch.stack(audio_inputs),
            "audio_masks": torch.stack(audio_masks),
            # labels
            "targets": {
                    "M": torch.tensor(targets_M, dtype=torch.float32),
                    "T": torch.tensor(targets_T, dtype=torch.float32),
                    "A": torch.tensor(targets_A, dtype=torch.float32)
                }
            }   
def collate_vce_custom_tav(batch):
    return {
        "text_tokens": torch.stack([b["text_tokens"] for b in batch], dim=0),
        "text_masks": torch.stack([b["text_masks"] for b in batch], dim=0),
        "text_context_tokens": torch.stack([b["text_context_tokens"] for b in batch], dim=0),
        "text_context_masks": torch.stack([b["text_context_masks"] for b in batch], dim=0),

        "audio_inputs": torch.stack([b["audio_inputs"] for b in batch], dim=0),
        "audio_masks": torch.stack([b["audio_masks"] for b in batch], dim=0),
        "audio_context_inputs": torch.stack([b["audio_context_inputs"] for b in batch], dim=0),
        "audio_context_masks": torch.stack([b["audio_context_masks"] for b in batch], dim=0),

        "video_pixel_values": torch.stack([b["video_pixel_values"] for b in batch], dim=0),  # [B,T,C,H,W]
        "targets": torch.stack([b["targets"] for b in batch], dim=0),  # [B]
    }

def data_loader(batch_size, dataset, modalities = "TA", text_context_length=2, audio_context_length=1):
    if dataset == 'mosi':
        csv_path = 'data/MOSI/label.csv'
        audio_file_path = "data/MOSI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset == 'mosei':
        csv_path = 'data/MOSEI/label.csv'
        audio_file_path = "data/MOSEI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset.lower() == 'vce_custom':
        audio_file_path = "data/VCE_CUSTOM/wav"
        if "V" in modalities:
            # use video cached pt files
            csv_path = "data/VCE_CUSTOM/labels_with_video_cache.csv"
            if "A" in modalities:
                # T+A+V (+ context) dataset
                train_data = Dataset_vce_custom_tav(
                    csv_path, audio_file_path, "train",
                    text_context_length=text_context_length,
                    audio_context_length=audio_context_length,
                )
                test_data = Dataset_vce_custom_tav(
                    csv_path, audio_file_path, "test",
                    text_context_length=text_context_length,
                    audio_context_length=audio_context_length,
                )
                val_data = Dataset_vce_custom_tav(
                    csv_path, audio_file_path, "valid",
                    text_context_length=text_context_length,
                    audio_context_length=audio_context_length)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_vce_custom_tav)
                test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_vce_custom_tav)
                val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_vce_custom_tav)
                return train_loader, test_loader, val_loader
            else:
                #only text and video, no audio context tensors
                df = pd.read_csv(csv_path)
                tokenizer = AutoTokenizer.from_pretrained("roberta-large")
                train_df = df[df["mode"] == "train"].reset_index(drop=True)
                test_df  = df[df["mode"] == "test"].reset_index(drop=True)
                val_df   = df[df["mode"] == "valid"].reset_index(drop=True)
                train_data = Dataset_vce_custom_text_video(train_df, tokenizer)
                test_data  = Dataset_vce_custom_text_video(test_df, tokenizer)
                val_data   = Dataset_vce_custom_text_video(val_df, tokenizer)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_vce_text_video)
                test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_vce_text_video)
                val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_vce_text_video)
                return train_loader, test_loader, val_loader
    else:
        csv_path = 'data/SIMS/label.csv'
        audio_file_path = "data/SIMS/wav"
        train_data = Dataset_sims(csv_path, audio_file_path, 'train')
        test_data = Dataset_sims(csv_path, audio_file_path, 'test')
        val_data = Dataset_sims(csv_path, audio_file_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        return train_loader, test_loader, val_loader
