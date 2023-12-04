import torch 
import torch.nn as nn 

class DepressionClassifier(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                       out_channels=16,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(in_channels=16,
                       out_channels=32,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(in_channels=32,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3)
        )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512, dropout=0.4, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        # Linear softmax layer
        self.out_linear = nn.Linear(320,num_targets)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self,
                pos_deep_embed: torch.Tensor,
                pos_mel_feat: torch.Tensor, 
                neg_deep_embed: torch.Tensor, 
                neg_mel_feat: torch.Tensor, 
                neutral_deep_embed: torch.Tensor, 
                neutral_mel_feat: torch.Tensor):
        # conv embedding
        pos_conv_embedding = self.conv2Dblock(pos_mel_feat) #(b,channel,freq,time)
        neg_conv_embedding = self.conv2Dblock(neg_mel_feat)
        neutral_conv_embedding = self.conv2Dblock(neg_mel_feat)
        # conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension
        pos_embed = torch.flatten(pos_conv_embedding, start_dim=1)
        neg_embed = torch.flatten(neg_conv_embedding, start_dim=1)
        neutral_embed = torch.flatten(neutral_conv_embedding, start_dim=1)
        # transformer embedding
        # x_reduced = self.transf_maxpool(x)
        # x_reduced = torch.squeeze(x_reduced,1)
        # x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        pos_transf_out = self.transf_encoder(pos_deep_embed)
        pos_transf_embedding = torch.mean(pos_transf_out, dim=0)
        neg_transf_out = self.transf_encoder(neg_deep_embed)
        neg_transf_embedding = torch.mean(neg_transf_out, dim=0)
        neutral_transf_out = self.transf_encoder(neutral_deep_embed)
        neutral_transf_embedding = torch.mean(neutral_transf_out, dim=0)
        # concatenate
        complete_embedding = torch.cat([pos_embed, neg_embed, neutral_embed,pos_transf_embedding,neg_transf_embedding,neutral_transf_embedding], dim=1) 
        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax