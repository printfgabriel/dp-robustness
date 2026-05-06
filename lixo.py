
# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
        
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.gn1 = nn.GroupNorm(8, 64)  # 8 grupos, 64 canais
        
#         self.layer1 = self._make_layer(64, 64, 2, stride=1)
#         self.layer2 = self._make_layer(64, 128, 2, stride=2)
#         self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def _make_layer(self, in_ch, out_ch, num_blocks, stride):
#         layers = [self._basic_block(in_ch, out_ch, stride)]
#         for _ in range(1, num_blocks):
#             layers.append(self._basic_block(out_ch, out_ch, 1))
#         return nn.Sequential(*layers)
    
#     # trocando BatchNorm por GroupNorm pra ser compatível com DP
#     def _basic_block(self, in_ch, out_ch, stride):
#         # Conv1
#         conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
#         gn1 = nn.GroupNorm(8, out_ch)
        
#         # Conv2
#         conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
#         gn2 = nn.GroupNorm(8, out_ch)
        
        
#         shortcut = nn.Sequential()
#         if stride != 1 or in_ch != out_ch:
#             shortcut = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
#                 nn.GroupNorm(8, out_ch)
#             )
        
#         return BasicBlock(conv1, gn1, conv2, gn2, shortcut)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.gn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)


# class BasicBlock(nn.Module):
#     def __init__(self, conv1, gn1, conv2, gn2, shortcut):
#         super().__init__()
#         self.conv1 = conv1
#         self.gn1 = gn1
#         self.conv2 = conv2
#         self.gn2 = gn2
#         self.shortcut = shortcut

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.gn2(self.conv2(out))
#         out = out + self.shortcut(identity)
#         return F.relu(out)