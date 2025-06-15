from transformers import LongformerModel
import torch
from torch import nn
import torchvision
from torch.nn import functional as F

class LongFormerVGGClassifier(nn.Module):

    def __init__(self, cfg, is_pre_vgg=True):
        super().__init__()

        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, cfg.common_hidden_dim)

        self.vgg = torchvision.models.vgg19(pretrained=True)
        if is_pre_vgg:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        self.image_fc = nn.Linear(1000, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim * 2, cfg.output_dim)
        
    def forward(self, text, img):

        text = text[:, :4096]

        attention_mask = torch.ones(text.shape, dtype=torch.long, device=text.device)
        global_attention_mask = torch.zeros(text.shape, dtype=torch.long, device=text.device)
        outputs = self.bert(text, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        text_features = self.text_fc(output)

        img = self.vgg(img)
        img_features = F.leaky_relu(self.image_fc(img))

        combine_features = torch.cat((text_features, img_features), 1)
        final_out = self.classifier(combine_features)

        final_out = torch.sigmoid(final_out)

        return final_out


class LongFormerClassifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(cfg.dropout)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, cfg.common_hidden_dim)

        self.classifier = nn.Linear(cfg.common_hidden_dim, cfg.output_dim)
        
    def forward(self, text, img=None):

        text = text[:, :4096]

        attention_mask = torch.ones(text.shape, dtype=torch.long, device=text.device)
        global_attention_mask = torch.zeros(text.shape, dtype=torch.long, device=text.device)
        outputs = self.bert(text, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        text_features = self.text_fc(output)

        final_out = self.classifier(text_features)

        final_out = torch.sigmoid(final_out)

        return final_out