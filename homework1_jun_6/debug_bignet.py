import torch
from homework.bignet import BigNet, BIGNET_DIM

# Set breakpoint on line below to see __init__
net = BigNet()

x = torch.randn(1, BIGNET_DIM)  # 1 sample, 1024 numbers

output = net(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
