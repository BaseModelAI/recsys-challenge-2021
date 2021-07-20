import torch
import torch.nn as nn
import torch.nn.functional as F 


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)


class Model(nn.Module):
    def __init__(self, hidden_size, num_languages, embedding_size, num_categorical_features, num_numerical_features, sketch_depth, sketch_width):
        super().__init__()
        input_dim =  num_categorical_features + 16 * num_numerical_features + embedding_size * 3 + sketch_depth*sketch_width
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, 4)
        self.projection = nn.Linear(input_dim, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.language = Embedding(num_languages+1, embedding_size)
        self.hour = Embedding(24, embedding_size)
        self.day = Embedding(31, embedding_size)


    def forward(self, x_input, language, twitt_hour, twitt_day):
        """
        Feed forward network with residual connections.
        """
        x_input = torch.cat((x_input, self.language(language), self.hour(twitt_hour), self.day(twitt_day)), axis=1)
        x_proj = self.projection(x_input)
        x_ = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
        x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        return x