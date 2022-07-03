
import torchvision.models as models
import torch.nn as nn
import copy
import torchvision.transforms as transforms


class PytorchVGG(nn.Module):
    def __init__(
        self, 
        fc=1, 
        conv=None, 
        av_pool=False
        ):
        super(PytorchVGG, self).__init__()

        self.basemodel = models.vgg16(pretrained=True)
        self.initiate()

        self.conv = conv
        self.fc = fc
        self.av_pool = av_pool

        if self.conv:
            if self.conv==3:
                self.features = list(self.basemodel.features.children()) [:17]
                self.s = 200704
                if self.av_pool:
                    self.features = self.features[:-1]
                    self.features.append(nn.AvgPool2d(6, stride=6, padding=3))
                    self.s = 25600

            elif self.conv==4:
                self.features = list(self.basemodel.features.children()) [:24]
                self.s = 100352
                if self.av_pool:
                    self.features = self.features[:-1]
                    self.features.append(nn.AvgPool2d(4, stride=4, padding=0))
                    self.s = 25088

            elif self.conv==5:
                self.features = list(self.basemodel.features.children()) 
                self.s = 25088
                if self.av_pool:
                    self.features = self.features[:-1]
                    self.features.append(nn.AvgPool2d(2, stride=2, padding=0))
                    self.s = 25088
            else:
                raise Exception(f"The Conv layer={self.conv} should be 3, 4, or 5") 
        
            self.myVGG = nn.Sequential(*self.features)

        elif self.fc:
            self.myVGG = copy.deepcopy(self.basemodel)
            if self.fc == 1:
                self.myVGG.classifier = self.myVGG.classifier[:3]
                self.s = 4096
            elif self.fc == 2:
                self.myVGG.classifier = self.myVGG.classifier[:-1]
                self.s = 4096
        else:
            raise Exception(f"at least one of these arguments, fc or conv, should be specified.") 


    def forward(
        self, 
        x
        ):
        x = self.myVGG(x)
        if len(x.shape)>2:
            x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return x

    def initiate(
        self
        ):
        self.basemodel.eval()
        for param in self.basemodel.parameters():
            param.requires_grad = False


def VGG16normalization(
    mean=None, 
    std=None
    ):
    
    if not mean:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    tra = [transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ]
    return tra

