from torch import nn
# import torch
# import torch.nn.functional as F

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.NUMBER_OF_FRAMES = 10
        self.conv_3d = nn.Sequential(
            nn.Conv3d(self.NUMBER_OF_FRAMES, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.ReLU(True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.ReLU(True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.ReLU(True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=2),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=38400, out_features=10),
            nn.BatchNorm1d(10),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input_tensor):
        input_tensor = self.conv_3d(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        logits = self.linear(input_tensor)
        return logits

# # class InceptionModule(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super(InceptionModule, self).__init__()
# #         self.b0  = nn.ReLU(nn.Conv3d(in_channels, out_channels[0], kernel_size=1, padding=1))
# #         self.b1a = nn.ReLU(nn.Conv3d(in_channels, out_channels[0], kernel_size=1, padding=1))
# #         self.b1b = nn.ReLU(nn.Conv3d(out_channels[0], out_channels[1], kernel_size=3, padding=1))
# #         self.b2a = nn.ReLU(nn.Conv3d(in_channels, out_channels[0], kernel_size=1, padding=1))
# #         self.b2b = nn.ReLU(nn.Conv3d(out_channels[0], out_channels[1], kernel_size=3, padding=1))
# #         self.b3a = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
# #         self.b3b = nn.ReLU(nn.Conv3d(out_channels[0], out_channels[1], kernel_size=1, padding=1))
# #     def forward(self, x):
# #         b0 = self.b0(x)
# #         b1 = self.b1b(self.b1a(x))
# #         b2 = self.b2b(self.b2a(x))
# #         b3 = self.b3b(self.b3a(x))
# #         print(b0.shape, b1.shape, b2.shape, b3.shape)
# #         print("--")
# #         return torch.cat([b0,b1,b2,b3], dim=1)

# # class I3D(nn.Module):
# #     def __init__(self):
# #         super(I3D, self).__init__()
# #         self.NUMBER_OF_FRAMES = 8
# #         self.conv_3d = nn.Sequential(
# #             nn.Conv3d(self.NUMBER_OF_FRAMES, 64, kernel_size=7, stride=2, padding=2),
# #             nn.BatchNorm3d(64, eps=0.001, momentum=0.01),
# #             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
# #             nn.ReLU(True),
# #             nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=2),
# #             nn.BatchNorm3d(128, eps=0.001, momentum=0.01),
# #             nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=2),
# #             nn.BatchNorm3d(128, eps=0.001, momentum=0.01),
# #             nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
# #             nn.ReLU(True),
# #             InceptionModule(128, [128, 256]),
# #             InceptionModule(256, [256, 512]),
# #             nn.MaxPool3d(kernel_size=3, stride=2),
# #             InceptionModule(512, [512, 512]),
# #             InceptionModule(512, [512, 512]),
# #             InceptionModule(512, [512, 512]),
# #             InceptionModule(512, [512, 512]),
# #             InceptionModule(512, [512, 512]),
# #             nn.MaxPool3d(kernel_size=3, stride=2),
# #             InceptionModule(512, [512, 512]),
# #             InceptionModule(512, [512, 512]),
# #             nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),
# #             nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=2),
# #         )
# #         self.linear = nn.Sequential(
# #             nn.Linear(in_features=185856, out_features=4096),
# #             nn.BatchNorm1d(4096, eps=0.001, momentum=0.01),
# #             nn.ReLU(True),
# #             nn.Linear(in_features=4096, out_features=2),
# #             nn.BatchNorm1d(256, eps=0.001, momentum=0.01),
# #             nn.ReLU(True),
# #             nn.Linear(in_features=256, out_features=2)
# #         )

# #     def forward(self, x):
# #         print("Before - ", x.shape)
# #         x = self.conv_3d(x)
# #         print("After Inception - ", x.shape)
# #         x = self.linear(x)
# #         print("After Linear - ", x.shape)
# #         return x


# class C3D(nn.Module):
#     def __init__(self):
#         super(C3D, self).__init__()
#         self.NUMBER_OF_FRAMES = 10
#         self.conv1 = nn.Conv3d(self.NUMBER_OF_FRAMES, 64, kernel_size=(3, 3, 3), stride=1 ,padding=(3, 3, 3))
#         self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))

#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1 ,padding=(2, 2, 2))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

#         self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1,  padding=(2, 2, 2))
#         self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1,  padding=(2, 2, 2))
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

#         self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=(2, 2, 2))
#         self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(2, 2, 2))
#         self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

#         self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(2, 2, 2))
#         self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(2, 2, 2))
#         self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

#         self.fc6 = nn.Linear(131072, 10)
#         self.fc8 = nn.Linear(10, 2)

#         self.dropout = nn.Dropout(p=0.5)

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # print("First - ", x.shape)
#         h = self.relu(self.conv1(x))
#         h = self.pool1(h)

#         # print("After Pool 1 - ", h.shape)
#         h = self.relu(self.conv2(h))
#         h = self.pool2(h)

#         # print("After Pool 2 - ", h.shape)
#         h = self.relu(self.conv3a(h))
#         h = self.relu(self.conv3b(h))
#         # print("In Pool 2 - ", h.shape)
#         h = self.pool3(h)

#         # print("After Pool 3 - ", h.shape)
#         h = self.relu(self.conv4a(h))
#         h = self.relu(self.conv4b(h))
#         h = self.pool4(h)

#         # print("After Pool 4 - ", h.shape)
#         h = self.relu(self.conv5a(h))
#         h = self.relu(self.conv5b(h))
#         h = self.pool5(h)

#         h = h.view(h.size(0), -1)
#         h = self.relu(self.fc6(h))
#         h = self.dropout(h)

#         logits = self.fc8(h)

#         return logits