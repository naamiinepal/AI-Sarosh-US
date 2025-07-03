import torch
import torch.nn as nn
from pytorchvideo.models.hub import mvit_base_32x3

class MViTEncoder(nn.Module):
    def __init__(self, cls_layer_idx=15):
        super().__init__()
        self.backbone = mvit_base_32x3(pretrained=True)
        self.backbone.head.proj = nn.Identity()
        self.cls_layer_idx = cls_layer_idx
        self.cls_embedding = None

        # Hook to extract CLS token embedding
        def hook_fn(module, input, output):
            # output shape: (B, N, C) — CLS token is at index 0
            self.cls_embedding = output[0][:, 0, :]

        self.backbone.blocks[self.cls_layer_idx].register_forward_hook(hook_fn)
        
        

    def forward(self, x):
        x = x.permute((0, 2, 1, 3, 4))  # (B, C, T, H, W)
        features = self.backbone(x)
        return features, self.cls_embedding

class MViTHead(nn.Module):
    def __init__(self, d_embed, intermediate_dims, num_classes):
        super(MViTHead, self).__init__()
        layers = []
        if (num_classes == 2):
            final_nodes = 1
        else:
            final_nodes = num_classes
        if len(intermediate_dims) == 0:
            layers.append(nn.Linear(d_embed, num_classes))
        else:
            # First layer: from embedding to first intermediate dim
            layers.append(nn.Linear(d_embed, intermediate_dims[0]))

            # Optional hidden layers if intermediate_dims has more than one value
            for in_dim, out_dim in zip(intermediate_dims[:-1], intermediate_dims[1:]):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(in_dim, out_dim))

            # Final layer: last intermediate dim to num_classes
            layers.append(nn.ReLU())
            layers.append(nn.Linear(intermediate_dims[-1], final_nodes))

        self.fcs = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcs(x)
    

class MViTModel(nn.Module):
    def __init__(self, num_classes=2, cls_layer_idx=15, use_head_only=False, use_head=True, d_embed=768, intermediate_dims=[256]):
        super().__init__()
        self.encoder = MViTEncoder(cls_layer_idx=cls_layer_idx)
        self.head = MViTHead(d_embed=768, intermediate_dims=[256], num_classes=num_classes) 
        self.use_head_only = use_head_only
        self.use_head = use_head

    def forward(self, x):
        
        if self.use_head_only == True and self.use_head == False:
            return "Errrror"
        
        if self.use_head_only:
            return self.head(x)
        
        
        features, cls_token = self.encoder(x)
        
        if self.use_head:
            out = features

        out = self.head(features)


        return out, cls_token
    