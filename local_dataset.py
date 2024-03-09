#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

from glob import glob

import torch
import torchaudio
import torch.nn.functional as F
import librosa

import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2Processor

from IPython.display import Audio, display, Markdown

class AudioEmotionsDataset():
    def __init__(self, data_path=None, train_split=0.8, batch_size=64, max_size=None):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.data_path = data_path if data_path is not None else "data/audio-emotions"
        self.train_split = train_split
        
        # if path does not exist, download the dataset and extract
        if not os.path.exists(self.data_path):
            raise Exception(f"""
                Dataset path not found at {self.data_path}.
                Please download the dataset from https://www.kaggle.com/uldisvalainis/audio-emotions
                and save it in the relevant path.
                """)
        
        metadata = {
            "angry": sorted(glob(f"{self.data_path}/Angry/*.wav")),
            "sad": sorted(glob(f"{self.data_path}/Sad/*.wav")),
            "disgusted": sorted(glob(f"{self.data_path}/Disgusted/*.wav")),
            "fearful": sorted(glob(f"{self.data_path}/Fearful/*.wav")),
            "happy": sorted(glob(f"{self.data_path}/Happy/*.wav")),
            "neutral": sorted(glob(f"{self.data_path}/Neutral/*.wav")),
            "surprised": sorted(glob(f"{self.data_path}/Surprised/*.wav")),
        }

        train_meta = {
            "angry": metadata["angry"][:int(self.train_split * len(metadata["angry"]))],
            "sad": metadata["sad"][:int(self.train_split * len(metadata["sad"]))],
            "disgusted": metadata["disgusted"][:int(self.train_split * len(metadata["disgusted"]))],
            "fearful": metadata["fearful"][:int(self.train_split * len(metadata["fearful"]))],
            "happy": metadata["happy"][:int(self.train_split * len(metadata["happy"]))],
            "neutral": metadata["neutral"][:int(self.train_split * len(metadata["neutral"]))],
            "surprised": metadata["surprised"][:int(self.train_split * len(metadata["surprised"]))],
        }

        test_meta = {
            "angry": metadata["angry"][int(self.train_split * len(metadata["angry"])):],
            "sad": metadata["sad"][int(self.train_split * len(metadata["sad"])):],
            "disgusted": metadata["disgusted"][int(self.train_split * len(metadata["disgusted"])):],
            "fearful": metadata["fearful"][int(self.train_split * len(metadata["fearful"])):],
            "happy": metadata["happy"][int(self.train_split * len(metadata["happy"])):],
            "neutral": metadata["neutral"][int(self.train_split * len(metadata["neutral"])):],
            "surprised": metadata["surprised"][int(self.train_split * len(metadata["surprised"])):],
        }
        
        # limit to size
        if max_size is not None:
            for emotion in train_meta:
                train_meta[emotion] = train_meta[emotion][:max_size]
            for emotion in test_meta:
                test_meta[emotion] = test_meta[emotion][:max_size]
        
        waveforms_train, X_train, y_train = [], [], []
        
        # NOTE: changing the class map here will change the numerical values for the classes.
        self.class_map = {0: "angry", 1: "sad", 2: "disgusted", 3: "fearful", 4: "happy", 5: "neutral", 6: "surprised"}
        self.class_map_inv = {v: k for k, v in self.class_map.items()}
        
        for emotion in train_meta:
            for data_point in train_meta[emotion]:
                waveform, features = self.extract_features(data_point)
                waveforms_train.append(waveform)
                X_train.append(features)
                
            # y_train += [self.class_map_inv[emotion]] * len(train_meta[emotion])
            y_train += [self.one_hot_encode(self.class_map_inv[emotion])] * len(train_meta[emotion])
            
        waveforms_test, X_test, y_test = [], [], []
        for emotion in test_meta:
            for data_point in test_meta[emotion]:
                waveform, features = self.extract_features(data_point)
                waveforms_test.append(waveform)
                X_test.append(features)
                
            y_test += [self.one_hot_encode(self.class_map_inv[emotion])] * len(test_meta[emotion])
            
        # ZERO-PADDING
        # pad the features to the same length
        max_len = max([x.shape[1] for x in X_train + X_test])
        print(f"{max_len = }")
        
        for i in range(len(X_train)):
            m = nn.ZeroPad2d((0, max_len - X_train[i].shape[1]))
            X_train[i] = m(X_train[i]).squeeze(0)
            
        for i in range(len(X_test)):
            m = nn.ZeroPad2d((0, max_len - X_test[i].shape[1]))
            X_test[i] = m(X_test[i]).squeeze(0)
            
            
        # shuffle uniformly
        self.waveforms_train, self.X_train, self.y_train = waveforms_train, X_train, y_train
        self.waveforms_test, self.X_test, self.y_test = waveforms_test, X_test, y_test
        
        # zip train x and y
        zipped_train = list(zip(self.X_train, self.y_train))
        zipped_test = list(zip(self.X_test, self.y_test))
        
        self.train_dataloader = DataLoader(zipped_train, batch_size=batch_size, shuffle=True, num_workers=11, persistent_workers=True)
        self.test_dataloader = DataLoader(zipped_test, batch_size=batch_size, shuffle=True, num_workers=11, persistent_workers=True)
        
    @property
    def feature_count(self):
        """
            Get the number of features in the dataset.
        """
        return self.X_train[0].shape[0]
    
    @property
    def class_count(self):
        """
            Get the number of classes in the dataset.
        """
        return len(self.class_map)
    
    def extract_features(self, file: str):
        """
            Loads audio from a given file path and extracts features using the Wav2Vec2 model.
        """
        
        # NOTE: Wav2Vec2 was trained with a sampling rate of 16kHz,
        # so we need to resample the audio files to 16kHz.
        waveform, sample_rate = librosa.load(file, sr=16000)
        
        
        #? extract features
        features = self.feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        
        return waveform, features
    
    def shuffle_datapoints(self, waveforms, X, y):
        """
            Shuffles the data points uniformly.
        """
        zipped = list(zip(waveforms, X, y))
        random.shuffle(zipped)
        waveforms, X, y = zip(*zipped)
        return waveforms, X, y
    
    def transcribe(self, features):
        """
            (Useful for testing)
            Generate a transcription from `Wav2Vec2` features.
            
            Returns
            -------
            - word_offsets: list of dicts for each word with start and end times
            - transcription: the transcription of the audio
        """
        
        input_values = features # torch.tensor(features)
        logits = self.model(input_values).logits[0]
        predicted_ids = torch.argmax(logits, dim=-1)
        outputs = self.tokenizer.decode(predicted_ids, output_word_offsets=True)
        time_offset = self.model.config.inputs_to_logits_ratio / self.feature_extractor.sampling_rate
        
        word_offsets = [
            { "word": d["word"],
                "start_time": round(d["start_offset"] * time_offset, 2),
                "end_time": round(d["end_offset"] * time_offset, 2),
            }
            for d in outputs.word_offsets
        ]
        
        transcription = " ".join([ item['word'] for item in word_offsets])

        return word_offsets, transcription
    
    def one_hot_encode(self, y):
        """
            One-hot encode a single datapoint.
            (Maybe do this all at once for the entire dataset?)
        """
        return F.one_hot(torch.tensor(y), num_classes=len(self.class_map)).float()

    def len_train(self):
        """
            Get the length of the training split of the dataset.
        """
        return len(self.y_train)
    
    def len_test(self):
        """
            get the length of the testing split of the dataset.
        """
        return len(self.y_test)
    
    def get_train(self, idx):
        """
            Get a single training datapoint.
            
            if iterating, try accessing `dataset.X_train` and `dataset.y_train` directly.
        """
        return self.waveforms_train[idx], self.X_train[idx], self.y_train[idx]
    
    def get_test(self, idx):
        """
            Get a single testing datapoint.
            
            If iterating, try accessing `dataset.X_test` and `dataset.y_test` directly.
        """
        return self.waveforms_test[idx], self.X_test[idx], self.y_test[idx]
    
    def label_to_num(self, emotion):
        """
            Encode a label (e.g. "angry") to a numerical value (e.g. 0)
        """
        return self.class_map_inv[emotion]
    
    def num_to_label(self, emotion):
        """
            Decode a numerical value (e.g. 0) to a label (e.g. "angry")
        """
        return self.class_map[emotion]
