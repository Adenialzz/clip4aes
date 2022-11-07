import torch
import clip

class MLP(torch.nn.Module):
    def __init__(self, in_len=512, out_len=10):
        super().__init__()
        self.fc = torch.nn.Linear(in_len, out_len)

    def forward(self, x):
        return self.fc(x)

class BLIP4AesFormer(torch.nn.Module):
    def __init__(self, device, out_len=10, fc_bias=False):
        super().__init__()
        from lavis.models import load_model_and_preprocess
        self.blip_model, preprocessors, _ = load_model_and_preprocess('blip_feature_extractor', model_type='base', is_eval=True, device=device)
        self.preprocess = preprocessors['eval']
        self.aes_fc = torch.nn.Linear(768, out_len, bias=fc_bias)

    def forward(self, x):
        feat = self.blip_model.extract_features({'image': x}, mode='image').image_embeds[:, 0]
        return self.aes_fc(feat)
            

class CLIP4AesFormer(torch.nn.Module):
    def __init__(self, arch, device, out_len=10, fc_bias=False):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(arch, device)
        self.aes_fc = torch.nn.Linear(list(self.clip_model.parameters())[-1].shape[0], out_len, bias=fc_bias)

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
