from funasr import infer
import pdb

p = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")

res = p("/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/t_1/positive_out.wav", batch_size_token=5000)
print(res)
pdb.set_trace()