from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures, audioVisualization
import matplotlib.pyplot as plt
import torch
# from pyhanlp import *
import jieba
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pdb

def audio_feature_extract(file_name):
    [Fs, x] = audioBasicIO.read_audio_file(file_name)
    # convert dual channel to mono
    x = x.mean(axis=1)
    # pdb.set_trace()
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
    plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

def audio_visual(file_name):
    pass

def whisper2text(file_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "Jingmiao/whisper-small-chinese_base"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
        # low_cpu_mem_usage=True,
    )

    sample = file_name

    result = pipe(sample)
    return result

def text_feature_extract(text):
    # pdb.set_trace()
    text = text['text']
    text = ''.join([x for x in text if x != ' '])
    # words = HanLP.segment(text)
    words = jieba.cut(text)
    num_charaters = len(text)
    word_list = '/'.join(words).split('/')
    num_words = len(word_list)
    pdb.set_trace()

if __name__ == '__main__':
    file_name = "/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/t_1/negative_out.wav"
    #
    # audio_feature_extract(file_name)
    curr_txt = whisper2text(file_name)
    text_feature_extract(curr_txt)