# from LLMs.pyAudioAnalysis import audioBasicIO
# from LLMs.pyAudioAnalysis import ShortTermFeatures, audioVisualization
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
        output_hidden_state=True
        # low_cpu_mem_usage=True,
    )

    sample = file_name

    result = pipe(sample)
    pdb.set_trace
    return result

def scope2text(file_name):
    model_id = 'damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1'
    processor = Preprocessor.from_pretrained(model_id)
    inference_pipeline = mc_pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_id)
    # embed = get_embedding(model, file_name)
    rec_result = inference_pipeline(audio_in=file_name)
    print(rec_result)
    pdb.set_trace()
# Let's see how to retrieve time steps for a model

# @deprecated
def wav2vec(file_name):
    inference_pipeline = mc_pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch')

    rec_result = inference_pipeline(audio_in=file_name)
    print(rec_result)
    pdb.set_trace()

def text_feature_extract(text):
    text = text['text']
    text = ''.join([x for x in text if x != ' '])
    words = jieba.cut(text)
    num_charaters = len(text)
    word_list = '/'.join(words).split('/')
    num_words = len(word_list)
    pdb.set_trace()

def funasr_api(file_name):
    '''
    Triggered funasr api, which is implemented by Ali-Damo, git repo: https://github.com/alibaba/FunASR.git 
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
            - embeddings: every sentence embedding
    '''
    api = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")

    results = api(file_name, batch_size_token=5000)
    result = results[0] # type of dict, with key, value, text_postprocessed, time_stamp, sentences, embeddings
    pdb.set_trace()


if __name__ == '__main__':
    file_name = "/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/t_1/positive_out.wav"
    #
    # audio_feature_extract(file_name)
    # curr_txt = whisper2text(file_name)
    # curr_txt = scope2text(file_name)
    # wav2vec(file_name)
    res = funasr_api(file_name=file_name)
    pdb.set_trace()
    text_feature_extract(curr_txt)