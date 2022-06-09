import torch
import torch.nn as nn
from misc.layers import activations
from models.backbone import get_backbone

class IWV(nn.Module):
    """This class is the model used to classify the source / target domain 
    for DomainNet."""
    def __init__(self, config):
        super(IWV, self).__init__()
        self.config = config
        # input layer projection
        self.backbone, self.backbone_model, params, in_features = get_backbone(config)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=config.backbone.avgpool)
        self.flatten = nn.Flatten()
        if config.backbone.trainable:
            self.backbone_params = nn.ParameterList(params)
        self.in_layer = nn.Linear(in_features, config.model.IWV.n_hidden)
        self.in_norm = nn.BatchNorm1d(config.model.IWV.n_hidden)
        self.phi = activations[config.model.IWV.activation]()
        self.layers = nn.ModuleList()
        # create a variable list of hidden layers
        for i in range(config.model.IWV.n_layers):
            self.layers.append(nn.Linear(config.model.IWV.n_hidden, config.model.IWV.n_hidden))
            self.layers.append(nn.BatchNorm1d(config.model.IWV.n_hidden))
            self.layers.append(activations[config.model.IWV.activation]())
            self.layers.append(nn.Dropout(p=config.model.IWV.dropout))
        self.adaptation_layer = nn.Sequential(
            nn.Linear(config.model.IWV.n_hidden, config.model.IWV.n_hidden//2),
            nn.BatchNorm1d(config.model.IWV.n_hidden//2),
            activations[config.model.IWV.activation](),
            nn.Dropout(p=config.model.IWV.dropout),
            nn.Linear(config.model.IWV.n_hidden//2, config.model.IWV.n_hidden//2),
            activations[config.model.IWV.activation]()
        )
        # output head for the domain classifier
        self.out_class = nn.Sequential(
            nn.Linear(config.model.IWV.n_hidden//2, 2) # ->2 outputs: source and target domain
        )
        # output head for the class classifier
        self.aux_class = nn.Sequential(
            nn.Linear(config.model.IWV.n_hidden//2, config.dataloader.DomainNet.num_classes) # -> number of classes
        )
    
    def forward(self, x, *args):
        if self.train and self.config.backbone.trainable:
            self.backbone_model.train()
        else:
            self.backbone_model.eval()
        h = self.backbone(x)
        if self.config.backbone.apply_avgpool:
            h = self.avgpool(h)
        h = self.flatten(h)
        h = self.in_layer(h)
        h = self.in_norm(h)
        h = self.phi(h)
        for layer in self.layers:
            h = layer(h)
        h = self.adaptation_layer(h)
        c = self.out_class(h)
        a = self.aux_class(h)
        return c, a # 1st return value is the domain classification result, second is auxiliary class classification
