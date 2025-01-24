import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels, num_layers, num_first_filters):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.channel_reducers = nn.ModuleList()

        num_filters = num_first_filters
        encoder_filters = []

        # Contractive path
        for i in range(num_layers):
            layers = [
                nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ]
            if i > 0:  # MaxPooling from the second layer onwards
                layers.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(nn.Sequential(*layers))
            encoder_filters.append(num_filters)
            input_channels = num_filters
            num_filters *= 2

        # Bottleneck
        bottleneck_channels = num_filters // 2
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Expansive path
        for i in range(num_layers):
            num_filters //= 2
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channels, num_filters, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            ))
            bottleneck_channels = num_filters
            # Channel reducer after concatenation of encoder and decoder outputs
            self.channel_reducers.append(
                nn.Conv2d(encoder_filters[-(i + 1)] + num_filters, num_filters, kernel_size=1)
            )

        # Final Convolutional Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_first_filters, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_outputs = []

        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, layer in enumerate(self.decoder):
            # Upsample decoder output
            x = layer(x)

            # Get the corresponding encoder output
            enc_output = enc_outputs[-(i + 1)]

            # Resize decoder output to match encoder dimensions
            x = F.interpolate(x, size=enc_output.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate encoder and decoder outputs
            x = torch.cat((enc_output, x), dim=1)

            # Reduce channels after concatenation
            x = self.channel_reducers[i](x)

        # Final Convolutional Layer
        x = self.final_conv(x)

        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, input_channels, num_layers, num_first_filters):
        super(AttentionUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.channel_reducers = nn.ModuleList()

        num_filters = num_first_filters
        encoder_filters = []

        # Contractive path
        for i in range(num_layers):
            layers = [
                nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ]
            if i > 0:  # MaxPooling from the second layer onwards
                layers.insert(0, nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(nn.Sequential(*layers))
            encoder_filters.append(num_filters)
            input_channels = num_filters
            num_filters *= 2

        # Bottleneck
        bottleneck_channels = num_filters // 2
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # Expansive path
        for i in range(num_layers):
            num_filters //= 2
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(bottleneck_channels, num_filters, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            ))
            bottleneck_channels = num_filters

            # Attention block
            self.attention_blocks.append(
                Attention_block(F_g=num_filters, F_l=encoder_filters[-(i + 1)], F_int=num_filters // 2)
            )

            # Channel reducer after concatenation
            self.channel_reducers.append(
                nn.Sequential(
                    nn.Conv2d(encoder_filters[-(i + 1)] + num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(inplace=True)
                )
            )

        # Final Convolutional Layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_first_filters, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_outputs = []

        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, layer in enumerate(self.decoder):
            # Upsample decoder output
            x = layer(x)

            # Get the corresponding encoder output
            enc_output = enc_outputs[-(i + 1)]

            # Apply attention block
            attn_output = self.attention_blocks[i](g=x, x=enc_output)

            # Concatenate attention output and decoder output
            x = torch.cat((attn_output, x), dim=1)

            # Reduce channels after concatenation
            x = self.channel_reducers[i](x)

        # Final Convolutional Layer
        x = self.final_conv(x)

        return x