import torch
import clip

class MLP(torch.nn.Module):
    def __init__(self, in_len=512, out_len=10):
        super().__init__()
        self.fc = torch.nn.Linear(in_len, out_len)

    def forward(self, x):
        return self.fc(x)

class CLIP4AesFormer(torch.nn.Module):
    def __init__(self, arch, device, out_len=10):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(arch, device)
        self.aes_fc = torch.nn.Linear(list(self.clip_model.parameters())[-1].shape[0], out_len)

    def freeze_feats(self):
        for params in self.clip_model.visual.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = x.type(self.clip_model.dtype)
        feat = self.clip_model.visual(x)
        feat = feat.type(torch.float32)
        return self.aes_fc(feat)
            
        



if __name__ == '__main__':
    model = CLIP4AesFormer('ViT-B/32', 'cpu')
    from PIL import Image
    img = model.preprocess(Image.open("tools/CLIP.png")).unsqueeze(0).to('cpu')
    print(model(img).shape)
