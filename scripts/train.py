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
sys.path.insert(1,"./")
from get_voxels import collate_batch
from networks import EncoderCNN, DecoderRNN, UNet3D, Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

if "models" not in os.listdir("../"):
    os.mkdir("../models")
out_dir = "../models/"

if "reports" not in os.listdir("../"):
    os.mkdir("../reports")
log_file = open(os.path.join("../reports/log.txt"), "w")

smiles = []
with open("../datasets/raw/zinc15_druglike_clean_canonical_max60.smi") as f:
    i=0
    for i, line in enumerate(f):
        smiles.append(line[:-1])
        if i > 20000000:
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
encoder = Encoder()
net = UNet3D(5,5)
encoderCNN.to(device)
decoderRNN.to(device)
net.to(device)
encoder.to(device)

#Encoder Optimizer
criterionEncoder = nn.BCELoss()
#Encoder optimizer
optimizerEncoder = torch.optim.Adam(encoder.parameters(), lr = 0.001)

# Our loss finction
criterionNet = nn.BCELoss()
# The optimizer
optimizerNet = torch.optim.Adam(net.parameters(), lr=0.001) 

# Caption optimizer
criterionCaption = nn.CrossEntropyLoss()
caption_params = list(decoderRNN.parameters()) + list(encoderCNN.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

#Other training stuff
train_dataloader = DataLoader(smile_DS, batch_size=128, collate_fn=collate_batch)
scheduler = DDPMScheduler(num_train_timesteps=1000)


for i, (x, captions, pharm, lengths) in enumerate(train_dataloader):
    if (i+1) % 10 == 0:
        print("Batch {} of {}.".format(i+1,np.int64(np.ceil(len(train_dataloader.dataset)/train_dataloader.batch_size))))

    #Train Encoder and Unet
    ##Unet
    timesteps = torch.randint(
        0,
        scheduler.num_train_timesteps,
        (x.shape[0],),
        device=x.device,
    ).long()

    noise = torch.randn(x.shape).to(x.device)
    noisy_x = scheduler.add_noise(x, noise, timesteps)    
    noisy_x = noisy_x.type(torch.FloatTensor).to(device)
    x = x.to(device)
    pred = net(noisy_x)
    net_loss = criterionNet(pred, x)
    optimizerNet.zero_grad()
    net_loss.backward()
    optimizerNet.step()
    net_loss = net_loss.cpu()

    ##Encoder
    pharm = pharm.to(device)
    # Forward pass
    encoded_tensor = encoder(x)
    enc_loss=criterionEncoder(encoded_tensor, pharm)
    # Backward and optimize
    optimizerEncoder.zero_grad()
    enc_loss.backward()
    optimizerEncoder.step()


    ##Train Captioning Networks after ~6000 batches compounds 
    if i > 4000:
        captions = Variable(captions.to(device))
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        decoderRNN.zero_grad()
        encoderCNN.zero_grad()
        features = encoderCNN(pred.detach())
        outputs = decoderRNN(features, captions, lengths)
        cap_loss = criterionCaption(outputs, targets)
        cap_loss.backward()
        caption_optimizer.step()

    if (i+1)%60000 == 0:
        log_file.write("reducing learning rate/n")
        log_file.flush()        
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr

    if (i + 1) % 500 == 0:
        torch.save(net.state_dict(),"../models/net_weights_{}.pkl".format(i+1))
        torch.save(encoder.state_dict(),"../models/net_weights_{}.pkl".format(i+1))
        torch.save(encoderCNN.state_dict(),"../models/net_weights_{}.pkl".format(i+1))
        torch.save(decoderRNN.state_dict(),"../models/net_weights_{}.pkl".format(i+1))
        if i > 4000:
            log_file.write("Net Loss: {}\nEncoder Loss: {}\nCaptioning Loss: {}.\n".format(net_loss,enc_loss,cap_loss))
            log_file.flush()
        else:
            print("Net Loss: {}\nEncoder Loss: {}.\n".format(net_loss,enc_loss))

    if i == 210000:
        # We are Done!
        log_file.close()
        break