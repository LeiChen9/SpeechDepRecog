from torch.utils.data import Dataset, DataLoader 
import yaml 
from data_utils import full_feat_extract
import pdb

def custom_collate(batch):
    keys, pos_deep_embed, pos_mel_feat, neg_deep_embed, neg_mel_feat, neutral_deep_embed, neutral_mel_feat, label = zip(*batch)
    batch = dict(
        pos_deep_embed=pos_deep_embed,
        pos_mel_feat=pos_mel_feat,
        neg_deep_embed=neg_deep_embed,
        neg_mel_feat=neg_mel_feat,
        neutral_deep_embed=neutral_deep_embed,
        neutral_mel_feat=neutral_mel_feat,
        label=label
    )
    return keys, batch


class SDRDataset(Dataset):
    def __init__(self, data_config_file):
        with open(data_config_file, 'r') as f:
            self.data_config = yaml.safe_load(f)
        self.samples = []
        for key, value in self.data_config.items():
            value['key'] = key
            self.samples.append(value)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        curr_dic = self.samples[idx]
        # get key
        key = curr_dic['key']
        # pos wave
        pos_dic = full_feat_extract(curr_dic['pos'])
        pos_deep_embed, pos_mel_feat = pos_dic['deep_embed'], pos_dic['mel_feat']
        # neg wave 
        neg_dic = full_feat_extract(curr_dic['neg'])
        neg_deep_embed, neg_mel_feat = neg_dic['deep_embed'], neg_dic['mel_feat']
        # neutral wave
        neutral_dic = full_feat_extract(curr_dic['neutral'])
        neutral_deep_embed, neutral_mel_feat = neutral_dic['deep_embed'], neutral_dic['mel_feat']
        # load label
        with open(curr_dic['label'], 'r') as f:
            score = f.read()
            score = float(score)
            if score >= 53:
                label = 1
            else:
                label = 0
        return key, pos_deep_embed, pos_mel_feat, neg_deep_embed, neg_mel_feat, neutral_deep_embed, neutral_mel_feat, label

def dataloader_gen(dataset, batch_size, shuffle=True, num_workers=8):
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=custom_collate,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    data_config_file = "./configs/data_config.yaml"
    data_set = SDRDataset(data_config_file)
    data_loader = dataloader_gen(data_set, 4)
    for batch in data_loader:
        pdb.set_trace()