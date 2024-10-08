import torch.nn as nn

class MLP(nn.Module):
   def __init__(self):
      super(MLP, self).__init__()   
      self.model = nn.Sequential(
         # https://www.kaggle.com/code/sarahibrahim97/simple-neural-network-using-pytorch
         nn.LazyLinear(128),
         nn.PReLU(),
         nn.Dropout(0.2),

         nn.Linear(128, 64),
         nn.PReLU(),
         nn.BatchNorm1d(64),

         nn.Linear(64, 32),
         nn.PReLU(),

         nn.Linear(32, 1)
      )
   
   def forward(self, x): 
      return self.model(x)
