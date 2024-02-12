import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from diffusers import DDPMScheduler
import os
import time
import argparse
import math
import sys
sys.path.insert(1,"../scripts")
from get_voxels import collate_batch
from networks import EncoderCNN, DecoderRNN, UNet3D_tcond, Encoder


## Parse Arguments for restarting training

my_parser = argparse.ArgumentParser(description='Train diffusion molecule generator with predicted pharmacophore conditioned sampling')
my_parser.add_argument('--restart', action='store', type=bool, default=False, required=False, help="Restarting from interrupted training?")
my_parser.add_argument('--prev', action='store', type=int, default=None, required=False, help="Final batch number from previous run")

args = my_parser.parse_args()


#Grab Start Time
start_time = time.time()

#Set up models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

if "models" not in os.listdir("../"):
    os.mkdir("../models")
out_dir = "../models/"

if "reports" not in os.listdir("../"):
    os.mkdir("../reports")

log_file = open(os.path.join("../reports/log.txt"), "w")

log_file.write("Using device: {}\n".format(device))
log_file.flush()


smiles = []
with open("../datasets/raw/zinc15_30000000.smi") as f:
    i=0
    for i, line in enumerate(f):
        smiles.append(line[:-1])
        if i > 30000000:
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

batch_size = 128

# Define the networks
encoderCNN = EncoderCNN(5)
decoderRNN = DecoderRNN(512, 1024, 29, 1)
encoder = Encoder()
net = UNet3D_tcond(5,5)

###############

encoderCNN.to(device)
decoderRNN.to(device)
net.to(device)
encoder.to(device)

### Restarting stuff
start_i = 0
lr_cap = 0.001

if args.restart == True:
    assert args.prev != None
    if args.prev <= 6000:
        net.load_state_dict(torch.load("../models/net_weights_"+str(args.prev)+".pkl"))
        encoder.load_state_dict(torch.load("../models/encoder_weights_"+str(args.prev)+".pkl"))
        smiles = smiles[args.prev*batch_size:]
        start_i = args.prev
    else:
        net.load_state_dict(torch.load("../models/net_weights_"+str(args.prev)+".pkl"))
        encoder.load_state_dict(torch.load("../models/encoder_weights_"+str(args.prev)+".pkl"))
        encoderCNN.load_state_dict(torch.load("../models/encoderCNN_weights_"+str(args.prev)+".pkl"))
        decoderRNN.load_state_dict(torch.load("../models/decoderRNN_weights_"+str(args.prev)+".pkl"))
        smiles = smiles[args.prev*batch_size:]
        start_i = args.prev
        lr_cap = lr_cap/(2**(math.floor(args.prev/40000)))


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
caption_optimizer = torch.optim.Adam(caption_params, lr=lr_cap)


#Other training stuff
train_dataloader = DataLoader(smile_DS, batch_size=batch_size, collate_fn=collate_batch)
scheduler = DDPMScheduler(num_train_timesteps=1000)

for i, (x, captions, pharm, lengths) in enumerate(train_dataloader):

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
    pred = net(noisy_x, timesteps.to(device))
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
    if i+start_i > 6000:
        captions = Variable(captions.to(device))
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        noisy_x = scheduler.add_noise(x, noise.to(device), torch.stack([scheduler.timesteps[999]]*x.shape[0]).to(device))    
        pred = net(noisy_x.to(device),torch.stack([scheduler.timesteps[999]]*x.shape[0]).to(device)).detach()
        decoderRNN.zero_grad()
        encoderCNN.zero_grad()
        features = encoderCNN(pred)
        outputs = decoderRNN(features, captions, lengths)
        cap_loss = criterionCaption(outputs, targets)
        cap_loss.backward()
        caption_optimizer.step()
        cap_loss = cap_loss.cpu()

    if (i+start_i+1)%40000 == 0:
        log_file.write("reducing learning rate/n")
        log_file.flush()        
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr

    if (i+start_i+1) % 500 == 0:
        if (i+start_i) <= 6000:
            if (i+start_i+1)%4000 == 0:
                torch.save(net.state_dict(),"../models/net_weights_{}.pkl".format(i+start_i+1))
                torch.save(encoder.state_dict(),"../models/encoder_weights_{}.pkl".format(i+start_i+1))

            log_file.write("Batch: {}\nNet Loss: {}\nEncoder Loss: {}\n".format((i+start_i+1),net_loss,enc_loss))
            log_file.flush()

        else:
            if (i+start_i+1)%4000 == 0:
                torch.save(net.state_dict(),"../models/net_weights_{}.pkl".format(i+start_i+1))
                torch.save(encoder.state_dict(),"../models/encoder_weights_{}.pkl".format(i+start_i+1))
                torch.save(encoderCNN.state_dict(),"../models/encoderCNN_weights_{}.pkl".format(i+start_i+1))
                torch.save(decoderRNN.state_dict(),"../models/decoderRNN_weights_{}.pkl".format(i+start_i+1))
            log_file.write("Batch: {}\nNet Loss: {}\nEncoder Loss: {}\nCaptioning Loss: {}.\n".format((i+start_i+1),net_loss,enc_loss,cap_loss))
            log_file.flush()


    if time.time()-start_time > 255600:
        torch.save(net.state_dict(),"../models/net_weights_{}.pkl".format(i+start_i+1))
        torch.save(encoder.state_dict(),"../models/encoder_weights_{}.pkl".format(i+start_i+1))
        torch.save(encoderCNN.state_dict(),"../models/encoderCNN_weights_{}.pkl".format(i+start_i+1))
        torch.save(decoderRNN.state_dict(),"../models/decoderRNN_weights_{}.pkl".format(i+start_i+1))        
        log_file.write("Batch: {}\nNet Loss: {}\nEncoder Loss: {}\nCaptioning Loss: {}.\n".format((i+start_i+1),net_loss,enc_loss,cap_loss))
        break
