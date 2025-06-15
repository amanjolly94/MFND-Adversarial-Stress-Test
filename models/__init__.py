import torch
from torch import nn
import torchvision
from torch.nn import functional as F

from .cnn import CNNVGGClassifier, CNNClassifier
from .lstm import BiLSTMVGGClassifier, BiLSTMClassifier
from .longformer import LongFormerVGGClassifier, LongFormerClassifier

def build_model(cfg, is_pre_vgg=True):

    if cfg.name == "LstmVgg":
        model = BiLSTMVGGClassifier(cfg, is_pre_vgg)

    elif cfg.name == "CnnVgg":
        model = CNNVGGClassifier(cfg, is_pre_vgg)

    elif cfg.name == "BertVgg":
        model = LongFormerVGGClassifier(cfg, is_pre_vgg)

    elif cfg.name == "Lstm":
        model = BiLSTMClassifier(cfg)

    elif cfg.name == "Cnn":
        model = CNNClassifier(cfg)

    elif cfg.name == "Bert":
        model = LongFormerClassifier(cfg)

    elif cfg.name == "Vgg":
        model = VGGClassifier(cfg)

    else:
        raise NotImplementedError
    
    return model

def build_text_model(cfg):
    if cfg.name == "LstmVgg":
        model = BiLSTMClassifier(cfg)

    elif cfg.name == "CnnVgg":
        model = CNNClassifier(cfg)

    elif cfg.name == "BertVgg":
        model = LongFormerClassifier(cfg)

    else:
        raise NotImplementedError
    
    return model

def build_img_model(cfg, is_pre_vgg=True):
    model = VGGClassifier(cfg, is_pre_vgg)

    return model


class VGGClassifier(nn.Module):

    def __init__(self, cfg, is_pre_vgg=True):
        super().__init__()

        self.vgg = torchvision.models.vgg19(pretrained=True)
        if is_pre_vgg:
            for param in self.vgg.parameters():
                param.requires_grad = False
        
        self.image_fc = nn.Linear(1000, cfg.common_hidden_dim)
        self.classifier = nn.Linear(cfg.common_hidden_dim, cfg.output_dim)

    def forward(self, img, text=None):

        img = self.vgg(img)
        img_features = F.leaky_relu(self.image_fc(img))
        final_out = self.classifier(img_features)

        final_out = torch.sigmoid(final_out)

        return final_out