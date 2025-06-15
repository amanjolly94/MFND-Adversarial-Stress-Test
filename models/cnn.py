import torch
from torch import nn
import torchvision
from torch.nn import functional as F

class CNNVGGClassifier(nn.Module):

    def __init__(self, cfg, is_pre_vgg=True):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.embedding_dim, 
                      out_channels=cfg.n_filters, 
                      kernel_size=fs)
            for fs in cfg.filter_sizes
        ])

        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(len(cfg.filter_sizes) * cfg.n_filters, cfg.common_hidden_dim)

        self.vgg = torchvision.models.vgg19(pretrained=True)
        if is_pre_vgg:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        self.image_fc = nn.Linear(1000, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim * 2, cfg.output_dim)
        

    def forward(self, text, img):

        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.leaky_relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        text_features = F.leaky_relu(self.text_fc(cat))

        img = self.vgg(img)
        img_features = F.leaky_relu(self.image_fc(img))

        combine_features = torch.cat((text_features, img_features), 1)
        final_out = self.classifier(combine_features)

        final_out = torch.sigmoid(final_out)

        return final_out


class CNNClassifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.embedding_dim, 
                      out_channels=cfg.n_filters, 
                      kernel_size=fs)
            for fs in cfg.filter_sizes
        ])

        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(len(cfg.filter_sizes) * cfg.n_filters, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim, cfg.output_dim)
        
    def forward(self, text, img=None):

        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.leaky_relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        text_features = F.leaky_relu(self.text_fc(cat))

        final_out = self.classifier(text_features)

        final_out = torch.sigmoid(final_out)

        return final_out