import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConditionalBatchNorm1d(nn.BatchNorm1d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.expand(size)
        # print(weight.shape)
        # print(output.shape)
        bias = bias.expand(size)
        return weight * output + bias


class LinearConditionalBatchNorm1d(ConditionalBatchNorm1d):

    def __init__(self, conditionDim, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(LinearConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Linear(conditionDim, num_features)
        self.biases = nn.Linear(conditionDim, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(LinearConditionalBatchNorm1d, self).forward(
            input, weight, bias)


# Define the supervised contrastive learning loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize the feature vectors
        features_normalized = nn.functional.normalize(features, dim=-1, p=2)

        # Calculate the similarity matrix
        similarity_matrix = torch.matmul(features_normalized, features_normalized.T) / self.temperature

        # Gather positive and negative pairs
        mask = labels.expand(labels.size(0), labels.size(0)).eq(labels.expand(labels.size(0), labels.size(0)).T)
        positives = similarity_matrix[mask].view(labels.size(0), -1)

        # Calculate the loss
        negatives = similarity_matrix[~mask].view(labels.size(0), -1)

        # Adjust the shape of positives and negatives
        positives = positives.unsqueeze(2)
        negatives = negatives.unsqueeze(0)

        # Concatenate positives and negatives along the third dimension
        logits = torch.cat([positives, negatives], dim=2)

        # Flatten the logits for cross-entropy loss
        logits = logits.view(-1, 2)

        # Generate target labels (0 for positives, 1 for negatives)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(labels.device)

        # Calculate the cross-entropy loss
        return nn.functional.cross_entropy(logits, targets)


def weights_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


class VAE_encoder(nn.Module):
    def __init__(self,
                 in_feature: int = 512,
                 out_feature: int = 128,
                 latent_dim: int = 128,
                 num_class: int = 100
                 ):
        super(VAE_encoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_class = num_class

        self.encoder = nn.Sequential(
            nn.Linear(self.in_feature, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(self.latent_dim * 2, self.out_feature)
        self.fc_var = nn.Linear(self.latent_dim * 2, self.out_feature)

        self.classifier = nn.Linear(self.out_feature, self.num_class)

        self.apply(weights_init)

    def encode(self, x):
        """
                Encodes the input by passing through the encoder network
                and returns the latent codes.
                :param input: (Tensor) Input tensor to encoder [N x C x H x W]
                :return: (Tensor) List of latent codes
                """

        x = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class VAE_decoder(nn.Module):
    def __init__(self,
                 in_feature: int = 512,
                 out_feature: int = 128,
                 class_dim: int = 768,
                 latent_dim: int = 128,
                 ):
        super(VAE_decoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_feature = in_feature
        self.class_dim = class_dim
        self.out_feature = out_feature

        self.linear1 = nn.Linear(self.in_feature, self.latent_dim * 2)
        self.conditional_batch_norm1 = LinearConditionalBatchNorm1d(conditionDim=class_dim,
                                                                    num_features=self.latent_dim * 2)
        self.linear2 = nn.Linear(self.latent_dim * 2, self.latent_dim * 2)
        self.conditional_batch_norm2 = LinearConditionalBatchNorm1d(conditionDim=class_dim,
                                                                    num_features=self.latent_dim * 2)
        self.linear3 = nn.Linear(self.latent_dim * 2, self.latent_dim * 2)
        self.conditional_batch_norm3 = LinearConditionalBatchNorm1d(conditionDim=class_dim,
                                                                    num_features=self.latent_dim * 2)
        self.Lrelu = nn.LeakyReLU()

        # self.semantic_transfer=nn.Linear(self.class_dim,self.out_feature)

        self.final_layer = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.out_feature))

        self.apply(weights_init)

    def forward(self, z, label):
        # label=self.semantic_transfer(label)
        # z = torch.cat((z,label),dim=1)
        z = self.linear1(z)
        z = self.conditional_batch_norm1(z, label)
        z = self.Lrelu(z)
        z = self.linear2(z)
        z = self.conditional_batch_norm2(z, label)
        z = self.Lrelu(z)
        z = self.linear3(z)
        z = self.conditional_batch_norm3(z, label)
        z = self.Lrelu(z)

        z = self.final_layer(z)

        return z


# Discriminator network
class Discriminator(nn.Module):
    def __init__(self,
                 in_feature: int = 512,
                 latent_dim: int = 128,
                 class_dim: int = 500,
                 output_cell=10):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.in_feature = in_feature
        self.class_dim = class_dim
        self.output_cell = output_cell

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_feature, self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, self.in_feature)
        )

        self.classifer = nn.Linear(self.latent_dim * 2, self.output_cell)
        self.discriminate = nn.Linear(self.in_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.apply(weights_init)

    def forward(self, image):
        hidden = self.feature_extractor(image)
        disc = self.sigmoid(self.discriminate(hidden))
        # classifier = self.softmax(self.classifer(hidden))
        return disc, hidden
