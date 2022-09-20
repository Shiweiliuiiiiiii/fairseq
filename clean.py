import os 
import sys
import torch



checkpoint = torch.load(sys.argv[1], map_location='cpu')
checkpoint = checkpoint['model']
torch.save(checkpoint, map_location='cpu')









