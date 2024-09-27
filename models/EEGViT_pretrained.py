import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_pretrained(nn.Module):
    def __init__(self, model_name, kernel1_y=32, kernel2_x=8, img_size_y=14):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, kernel1_y),
            stride=(1, kernel1_y),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)


        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129,img_size_y)})
        config.update({'patch_size': (kernel2_x,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(kernel2_x, 1), stride=(kernel2_x, 1), padding=(0,0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        
        return x
