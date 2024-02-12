import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from diffusers import DDPMScheduler
import os
import sys
sys.path.insert(1,"../scripts")
from get_voxels import collate_batch
from networks import EncoderCNN, DecoderRNN, UNet3D_tcond, Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

if "models" not in os.listdir("../"):
    os.mkdir("../models")
out_dir = "../models/"

if "reports" not in os.listdir("../"):
    os.mkdir("../reports")

log_file = open(os.path.join("../reports/log_cap2.txt"), "w")

smiles = []
with open("../datasets/raw/zinc15_30000000.smi") as f:
    i=0
    for i, line in enumerate(f):
        smiles.append(line[:-1])
        if i > 5000000:
            break

class CustomImageDataset(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        return smile

smile_DS = CustomImageDataset(smiles)

# Define the networks
encoderCNN = EncoderCNN(5)
decoderRNN = DecoderRNN(512, 1024, 29, 1)
net = UNet3D_tcond(5,5)

net.load_state_dict(torch.load("../models/net_weights_39000.pkl"))

net.to(device)
encoderCNN.to(device)
decoderRNN.to(device)

# Caption optimizer
criterionCaption = nn.CrossEntropyLoss()
caption_params = list(decoderRNN.parameters()) + list(encoderCNN.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

#Other training stuff
train_dataloader = DataLoader(smile_DS, batch_size=128, collate_fn=collate_batch)
scheduler = DDPMScheduler(num_train_timesteps=1000)

for i, (x, captions, pharm, lengths) in enumerate(train_dataloader):

    captions = Variable(captions.to(device))
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    noise = torch.randn(x.shape)

    pred = net(x.to(device),torch.stack([scheduler.timesteps[999]]*x.shape[0]).to(device)).detach()

    decoderRNN.zero_grad()
    encoderCNN.zero_grad()
    features = encoderCNN(pred)
    outputs = decoderRNN(features, captions, lengths)
    cap_loss = criterionCaption(outputs, targets)
    cap_loss.backward()
    caption_optimizer.step()

    if (i+1)%20000 == 0:
        log_file.write("reducing learning rate/n")
        log_file.flush()        
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr

    if (i + 1) % 50 == 0:
        if (i+1) % 1000 == 0:
            torch.save(encoderCNN.state_dict(),"../models/encoderCNN_weights2_{}.pkl".format(i+1))
            torch.save(decoderRNN.state_dict(),"../models/decoderRNN_weights2_{}.pkl".format(i+1))

        log_file.write("Batch: {}\nCaptioning Loss: {}.\n".format((i+1),cap_loss))
        log_file.flush()

    if i == 39060:
        # We are Done!
        log_file.close()
        break
