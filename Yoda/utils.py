import argparse
import os
import json
import numpy as np 
from scipy.io.wavfile import read
import torch

def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

def get_speech_encoder(speech_encoder,device=None,**kargs):
    if speech_encoder == "vec768l12":
        from vencoder.ContentVec768L12 import ContentVec768L12
        speech_encoder_object = ContentVec768L12(device = device)
    elif speech_encoder == "vec256l9":
        from vencoder.ContentVec256L9 import ContentVec256L9
        speech_encoder_object = ContentVec256L9(device = device)
    elif speech_encoder == "vec256l9-onnx":
        from vencoder.ContentVec256L9_Onnx import ContentVec256L9_Onnx
        speech_encoder_object = ContentVec256L9_Onnx(device = device)
    elif speech_encoder == "vec256l12-onnx":
        from vencoder.ContentVec256L12_Onnx import ContentVec256L12_Onnx
        speech_encoder_object = ContentVec256L12_Onnx(device = device)
    elif speech_encoder == "vec768l9-onnx":
        from vencoder.ContentVec768L9_Onnx import ContentVec768L9_Onnx
        speech_encoder_object = ContentVec768L9_Onnx(device = device)
    elif speech_encoder == "vec768l12-onnx":
        from vencoder.ContentVec768L12_Onnx import ContentVec768L12_Onnx
        speech_encoder_object = ContentVec768L12_Onnx(device = device)
    elif speech_encoder == "hubertsoft-onnx":
        from vencoder.HubertSoft_Onnx import HubertSoft_Onnx
        speech_encoder_object = HubertSoft_Onnx(device = device)
    elif speech_encoder == "hubertsoft":
        from vencoder.HubertSoft import HubertSoft
        speech_encoder_object = HubertSoft(device = device)
    elif speech_encoder == "whisper-ppg":
        from vencoder.WhisperPPG import WhisperPPG
        speech_encoder_object = WhisperPPG(device = device)
    elif speech_encoder == "cnhubertlarge":
        from vencoder.CNHubertLarge import CNHubertLarge
        speech_encoder_object = CNHubertLarge(device = device)
    elif speech_encoder == "dphubert":
        from vencoder.DPHubert import DPHubert
        speech_encoder_object = DPHubert(device = device)
    elif speech_encoder == "whisper-ppg-large":
        from vencoder.WhisperPPGLarge import WhisperPPGLarge
        speech_encoder_object = WhisperPPGLarge(device = device)
    elif speech_encoder == "wavlmbase+":
        from vencoder.WavLMBasePlus import WavLMBasePlus
        speech_encoder_object = WavLMBasePlus(device = device)
    else:
        raise Exception("Unknown speech encoder")
    return speech_encoder_object 

def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    # can parse config file and model from arguments
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                      help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path, infer_mode = False):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)
  hparams =HParams(**config) if not infer_mode else InferHParams(**config)
  return hparams

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def get(self,index):
    return self.__dict__.get(index)
  
class InferHParams(HParams):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = InferHParams(**v)
      self[k] = v

  def __getattr__(self,index):
    return self.get(index)
