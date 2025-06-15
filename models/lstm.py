import torch
from torch import nn
import torchvision
from torch.nn import functional as F

class BiLSTMVGGClassifier(nn.Module):

    def __init__(self, cfg, is_pre_vgg=True):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        self.lstm = nn.LSTM(
            cfg.embedding_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers, 
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(cfg.hidden_dim * 2, cfg.common_hidden_dim)

        self.vgg = torchvision.models.vgg19(pretrained=True)
        if is_pre_vgg:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        self.image_fc = nn.Linear(1000, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim * 2, cfg.output_dim)
        

    def forward(self, text, img):

        embedded = self.embedding(text)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        text_features = self.text_fc(hidden)

        img = self.vgg(img)
        img_features = F.leaky_relu(self.image_fc(img))

        combine_features = torch.cat((text_features, img_features), 1)
        final_out = self.classifier(combine_features)

        final_out = torch.sigmoid(final_out)

        return final_out

class BiLSTMClassifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        self.lstm = nn.LSTM(
            cfg.embedding_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers, 
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(cfg.hidden_dim * 2, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim, cfg.output_dim)
        

    def forward(self, text, img=None):

        embedded = self.embedding(text)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        text_features = self.text_fc(hidden)

        final_out = self.classifier(text_features)

        final_out = torch.sigmoid(final_out)

        return final_out