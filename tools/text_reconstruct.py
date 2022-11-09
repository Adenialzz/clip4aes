# clip-text-decoder is `distilgpt2` trained on coco-caption2014 dataset
# ref repo: https://github.com/fkodom/clip-text-decoder

import clip
import argparse
from functools import lru_cache
from clip_text_decoder.common import load_tokenizer
from clip_text_decoder.model import Decoder, DecoderInferenceModel

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--text-path', type=str, default='test_text.txt')
    parser.add_argument('--text-decoder-path', type=str, default='/media/song/clip-text-decoder-results/lightning_logs/2gpus_coco/checkpoints/epoch=9-step=3240.ckpt')
    cfg = parser.parse_args()
    return cfg

def main(cfg):
    model, _ = clip.load('ViT-B/32', device=cfg.device)
    with open(cfg.text_path, 'r') as f:
        lines = f.readlines()
    text = [line.strip() for line in lines]
    text_inputs = clip.tokenize(text).to(cfg.device)
    text_features = model.encode_text(text_inputs)
    # text_features /= text_features.norm(dim=1, keepdim=True)    # [bs, 512]

    get_tokenizer = lru_cache(load_tokenizer)

    language_model = 'distilgpt2'
    model = Decoder.load_from_checkpoint(cfg.text_decoder_path)
    decoder = DecoderInferenceModel(model=model, tokenizer=get_tokenizer(language_model))

    for i in range(len(text)):
        caption = decoder(text_features[i], 64, 1)
        print(f'result {i}: ')
        print('\t', 'Origin Text:', text[i])
        print('\t', 'Rected Text:', caption)


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)

