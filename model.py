import torch
from torch import nn

class Conv2dReLU(nn.Module):
    def __init__(self, in_chnl, out_chnl, kernel_size, padding):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_chnl, out_chnl, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding, 
                              padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, data):
        data = self.conv(data)
        data = self.relu(data)
        return data


class VDSR_Net(nn.Module):
    def __init__(self, in_channels=3 , out_channels=3, depth=10):
        super(VDSR_Net, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
            nn.ReLU(True),
        )
        
        depth -= 2
        # self.vdsr_blocks = list([Conv2dReLU(64, 64, (2*d)+1, d) for d in range(depth)])
        self.vdsr_blocks = [Conv2dReLU(64, 64, 3, 1) for d in range(depth)]
        self.vdsr_blocks = nn.Sequential(*self.vdsr_blocks) # Unpack all of them into nn.Squential
        self.output_block = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros")
    
    def forward(self, data):
        original = data
        
        data = self.input_block(data)
        data = self.vdsr_blocks(data)
        data = self.output_block(data)
        
        data = torch.add(data, original)
        return data


if __name__ == "__main__":
    # Checking if the model works as intended
    # Let's test the VDSR network
    vdsr_net = VDSR_Net()
    print("-"*40, "NETWORK ARCHITECTURE", "-"*40)
    print(vdsr_net)
    print("-"*105)
    print("Feeding test input to the model: A single grayscale image of size 128x128")
    test_input = torch.randn((1, 3, 128, 128))
    out = vdsr_net(test_input)
    print(f"Output shape {out.shape}")
