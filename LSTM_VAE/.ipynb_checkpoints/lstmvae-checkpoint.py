import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch.nn.functional as F
from encoder import *
from decoder import *
from reparam import *


class VAE(nn.Module):
    
    def __init__(self, sequence_length, number_of_features, hidden_size=100, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0,
                 n_epochs=5, dropout_rate=0, loss='MSELoss', print_every=1, clip=True, max_grad_norm=5):

        super(VAE, self).__init__()

        if torch.cuda.is_available():
            print("using CUDA")
            self.use_cuda = True
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
            


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               cuda=self.dtype)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        
        if self.use_cuda:
            self.cuda()


        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
       

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def forward(self, x):
        
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    
    def trainer(self, dataset, save = False):
        
        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)
        
        self.train()

        for i in range(self.n_epochs):
            print("Epoch: ",  i)

            
            epoch_loss = 0
            t = 0

            for t, X in enumerate(train_loader):


                X = X[0].permute(1,0,2)
                X=X[:,:,:].type(self.dtype)

                self.optimizer.zero_grad()
                
                x = Variable(X, requires_grad = True)

                x_decoded, _ = self(x)

                latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

                kl_loss = -1*0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                recon_loss = self.loss_fn(x_decoded, x.detach())


                loss=kl_loss + 1000*recon_loss

                loss.backward()

                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
                epoch_loss += loss.item()

                self.optimizer.step()

                if (t + 1) % self.print_every == 0:
                    print('Batch: ',(t + 1), " loss: ",loss.item() , " recon_loss: ",recon_loss.item()," KL loss: ", kl_loss.item())

            print('Average loss: ',(epoch_loss / t))


    def reconstruct(self, dataset, save = False):
       

        self.eval()

        loader = DataLoader(dataset = dataset,batch_size = self.batch_size,shuffle = False,drop_last=True) 
        
        with torch.no_grad():
            x_decoded = []

            for t, x in enumerate(loader):
                x = x[0].permute(1, 0, 2)

                x = Variable(x.type(self.dtype), requires_grad = False)
                
                x_decoded_each, _ = self(x)
                
                x_decoded_each=x_decoded_each.cpu().data.numpy()
                
                x_decoded.append(x_decoded_each)

            x_decoded = np.concatenate(x_decoded, axis=1)

            return x_decoded

    def transform(self, dataset, save = False):
        
        self.eval()

        loader = DataLoader(dataset = dataset,batch_size = self.batch_size,shuffle = False,drop_last=False)
        
        with torch.no_grad():
            z = []

            for t, x in enumerate(loader):
                x = x[0].permute(1, 0, 2)
                x = x.type(self.dtype)
              
                z.append(self.lmbd(
                    self.encoder(Variable(x, requires_grad = False)
                    )).cpu().data.numpy())

            z = np.concatenate(z, axis=0)
            
            return z

        
    def decode(self,z):

        j=[]
    
        for i in z:
            x=torch.from_numpy(i)
            x=x.float()
            x= Variable(x, requires_grad = False)
            print(x.shape)
            x=x.view(1,1)            
            
            y=self.decoder(x)
            j.append(y.detach().cpu().numpy())
            
        return j

    def save(self, file_name):
        """
        Save model parameters
        """
        PATH =  './' + file_name
        
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Load model parameters
        """
        
        self.load_state_dict(torch.load(PATH))