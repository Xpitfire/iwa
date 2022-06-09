import torch.nn as nn
from misc.layers import activations
from models.backbone import get_backbone
from models.reversal import GradientReversalLayer


class MLP(nn.Module):
    latent_features = {}

    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        # input layer projection
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(config.model.MLP.n_in, config.model.MLP.n_hidden)
        self.phi = activations[config.model.MLP.activation]()
        self.layers = nn.ModuleList()
        # create a variable list of hidden layers
        for i in range(config.model.MLP.n_layers):
            self.layers.append(nn.Linear(config.model.MLP.n_hidden, config.model.MLP.n_hidden))
            self.layers.append(activations[config.model.MLP.activation]())
            self.layers.append(nn.Dropout(p=config.model.MLP.dropout))
        self.adaptation_layer = nn.Sequential(
            nn.Linear(config.model.MLP.n_hidden, config.model.MLP.n_hidden // 2),
            activations[config.model.MLP.activation]()
        )
        # output head for the domain classifier
        self.domain_layer = nn.Sequential(
            nn.Linear(config.model.MLP.n_hidden // 2, 2)  # -> 2 outputs: source and target domain
        )
        # output head for the class classifier
        self.out_class = nn.Sequential(
            nn.Linear(config.model.MLP.n_hidden // 2, config.dataloader.Moon.n_classes)
        )

    def forward(self, x, *args):
        alpha = args[0] if len(args) == 1 else 0 # zero disables gradient from domain classifier in backbone
        h = self.flatten(x)
        h = self.in_layer(h)
        h = self.phi(h)
        for layer in self.layers:
            h = layer(h)
        h = self.adaptation_layer(h)

        c = self.out_class(h)

        d = GradientReversalLayer.apply(h, alpha)
        d = self.domain_layer(d)
        return c, d
