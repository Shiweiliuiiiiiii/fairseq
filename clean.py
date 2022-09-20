import os 
import sys
import torch



checkpoint = torch.load(sys.argv[1], map_location='cpu')
print(checkpoint.keys())










