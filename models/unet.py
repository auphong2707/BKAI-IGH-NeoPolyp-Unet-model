import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, num_layers, 
                 kernel_size, padding, dropout_rate, upsampling_method, device):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            ]
            if dropout_rate > 0:
                layers.insert(2, nn.Dropout2d(p=dropout_rate))
            return nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)
            self.encoders.append(conv_block(in_ch, out_ch))

        # Bottleneck
        self.bottleneck = conv_block(base_channels * (2 ** (num_layers - 1)), base_channels * (2 ** num_layers))

        # Decoder layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)

            if upsampling_method == "TransposedConv":
                self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            elif upsampling_method == "Interpolation":
                self.upconvs.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))

            self.decoders.append(conv_block(in_ch, out_ch))

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.to(device)

    def forward(self, x):
        enc_feats = []

        # Encoder forward
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward
        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_feats[-(i + 1)]], dim=1)
            x = self.decoders[i](x)

        return self.final_conv(x)


if __name__ == '__main__':
    from config import *
    # Example Usage
    model = UNet(IN_CHANNELS, OUT_CHANNELS, BASE_CHANNELS, NUM_LAYERS,
                 KERNEL_SIZE, PADDING, DROPOUT_RATE, UPSAMPLING_METHOD, DEVICE)

    # Check the output
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    output = model(x)
    print(output.shape)  # Should match (batch_size, out_channels, height, width)
