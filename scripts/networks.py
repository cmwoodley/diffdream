import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""
class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res



class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            self.sig1 = nn.Sigmoid()
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: 
            out = self.conv3(out)
            out = self.sig1(out)
        return out
        



class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Create the first conv3d layer
        self.conv3d1 = nn.Conv3d(in_channels=5, out_channels=8, kernel_size=(3,3,3), stride=(1,1,1),padding=1)
        self.fc1 = nn.Linear(in_features=24, out_features=24)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU()

        self.conv3d2 = nn.Conv3d(in_channels=8, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1),padding=1)
        # self.fc2 = nn.Linear(in_features=24, out_features=24)
        self.bn2 = nn.BatchNorm3d(3)
        self.sig1 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv3d1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        # x = self.relu1(x)
        x = self.sig1(x)
        x = self.conv3d2(x)
        # x = self.fc2(x)
        x = self.bn2(x)
        x = self.sig1(x)
        return x

#Encoder and Decoder from Ligdream

#######################
# Captioning Networks #
#######################


class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
