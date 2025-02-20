import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0] 
config_path = CURRENT_DIR.rsplit('/', 2)[0] # upper 2 level dir
sys.path.append(config_path)

from LLMs.pyAudioAnalysis.pyAudioAnalysis import audioBasicIO
from LLMs.pyAudioAnalysis.pyAudioAnalysis import ShortTermFeatures, audioVisualization
import matplotlib.pyplot as plt
import torch
# from pyhanlp import *
import jieba
from transformers import AutoTokenizer, AutoModelForCTC
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from modelscope.pipelines import pipeline as mc_pipeline
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks
from funasr import infer
import modelscope
import pdb

def audio_feature_extract(file_name: str) -> dict:
    '''
    Extract acoustic feature from audio file, implemented by a Greece researcher, git repo: https://github.com/tyiannak/pyAudioAnalysis.git
    Params:
        file_name: absoulte path of audio file to be processed, must in "wav" format
    Return:
        type of tuple of (feat_dict, feature matrix, feature names)
        dict:
            - key: feature name
            - value: numeric representation of feature
        feeature matrix: 68 * 740
        feature names: list of len 68
    '''
    [Fs, x] = audioBasicIO.read_audio_file(file_name)
    # convert dual channel to mono
    x = x.mean(axis=1)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    
    audio_dict = {}
    for i in range(len(f_names)):
        audio_dict[f_names[i]] = F[i].reshape(-1)
    return audio_dict, F, f_names

def audio_visual(F, f_names) -> None:
    plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
    plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1])
    plt.show()

def text_feature_extract(funasr_dict):
    '''
    Text statistical feature extraction base on funasr result 
    Params:
        funasr_dict: @funasr_api 
    Return:
        type of dict
        - key: feature name
        - value: feature number
    '''
    text = funasr_dict['value']
    word_list = [x for x in jieba.cut(text)]
    char_list = funasr_dict['text_postprocessed'].split(" ")
    num_words, num_chars = len(word_list), len(char_list)
    res_dict = {
        "num_of_words": num_words,
        "num_chars": num_chars
    }
    return res_dict

def funasr_api(file_name: str) -> dict:
    '''
    Trigger funasr api, work for Speech2Text, implemented by Ali-Damo, git repo: https://github.com/alibaba/FunASR.git 
    Params:
        file_name: absoulte path of audio file to be processed, must in "wav" format
    Return:
        type of dict
            - key: the index, which is the relative file_name (get rid of prefix)
            - value: transcripted text
            - text_postprocessed: text which is splitted
            - time_stamp: every word start time and end time, list of 2-element list.
            - sentences: list of every sentence
                - text: this sentence 
                - start: start time
                - end: end time
                - text_seg: text segement joined by " "
                - ts_list: time_stamp list
            - embeddings: every sentence embedding, torch tensor with 1 x word_len x 512
    '''
    api = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")

    results = api(file_name, batch_size_token=5000)
    result = results[0]
    return result


if __name__ == '__main__':
    file_name = "/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/t_1/positive_out.wav"
    # # get funasr feature
    funasr_dict = funasr_api(file_name=file_name)
    # # get audio feature
    # audio_dict, F, f_names = audio_feature_extract(file_name=file_name)
    # # audio visualization
    # audio_visual(F, f_names)
    # get word feat
    text_feature_extract(funasr_dict)