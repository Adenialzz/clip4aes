# clip-text-decoder is `distilgpt2` trained on coco-caption2014 dataset
# ref repo: https://github.com/fkodom/clip-text-decoder

import torch
import torch.nn.functional as F
import clip
import argparse
from functools import lru_cache
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

class GPT2Decoder:
    def __init__(self, language_model_name='distilgpt2', device='cuda'):
        assert language_model_name in ['gpt2', 'distilgpt2', 'gpt2medium']
        self.device = device
        self.language_model = self.load_language_model(language_model_name).to(self.device)
        self.tokenizer = self.load_tokenizer(language_model_name)

    def load_language_model(self, model_name):
        config = GPT2Config.from_pretrained(model_name, add_cross_attention=True)
        model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        return model

    @lru_cache
    def load_tokenizer(self, model_name):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_decoder_checkpoint(self, path):
        state_dict = torch.load(path)['state_dict']
        new_state_dict = {k[len('language_model.'): ]: v for k, v in state_dict.items()}

        self.language_model.load_state_dict(new_state_dict)

    def forward_decoder(self, input_ids, encoder_hidden_states, attention_mask=None, labels=None):
        batch_size, _, num_features = encoder_hidden_states.shape
        # TODO: Check if we can get '768' (num_features) from the GPT2 model.
        hidden = torch.zeros(
            size=(batch_size, 1, 768),
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        hidden[:, :, :num_features] = encoder_hidden_states

        return self.language_model(
            input_ids=input_ids,
            encoder_hidden_states=hidden,
            attention_mask=attention_mask,
            labels=labels,
        )

    
    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(self, x, max_len, beam_size):
        """Inference using beam search. For beam search, we predict one token per step.
        After each step, we keep only the 'beam_size' output sequences with the highest
        end-to-end confidence score. Repeat this process until at most 'max_len' tokens
        have been generated.
        """
        encoder_hidden_states = x.reshape(1, 1, -1).to(self.device)
        # Since we haven't performed any beam search steps yet, we just have one
        # set of input IDs (with a single "start" token). We use 'None' for the log
        # probability of this sequence, since it's not being predicted by the model.
        input_ids = [torch.tensor([self.tokenizer.bos_token_id], device=self.device)]
        beam_logprobs = None

        def _get_beam_outputs(_input_ids):
            """Performs inference on the 'input_ids' Tensor, and collects the top
            'beam_size' results by score. Returns a list of output Tensors, and
            their respective log-probabilities.
            """
            outputs = self.forward_decoder(_input_ids.unsqueeze(0), encoder_hidden_states)
            logits = outputs.logits[0, -1]
            logprobs = F.log_softmax(logits, dim=-1)

            topk_logprobs = logprobs.topk(k=beam_size)
            indices = topk_logprobs.indices
            logprobs = topk_logprobs.values
            output_ids = [
                torch.cat([_input_ids, idx.reshape(-1)], dim=0) for idx in indices
            ]

            return output_ids, logprobs

        for _ in range(max_len - 1):
            output_ids = []
            logprobs = []
            beams_done = []

            # Collect the top 'beam_size' results from each beam individually.
            for beam_idx, ids in enumerate(input_ids):
                # If 'beam_logprobs' is already defined, then we've predicted at least
                # one token already. And if the last token is equal to the "stop" token,
                # we don't need to perform inference with this beam anymore.
                if beam_logprobs and ids[-1].item() == self.tokenizer.eos_token_id:
                    output_ids.append(ids)
                    logprobs.append(beam_logprobs[beam_idx])
                    beams_done.append(True)
                    continue

                _output_ids, _logprobs = _get_beam_outputs(ids)
                if beam_logprobs is not None:
                    # Sum the log-probabilities of the existing beam and our predicted
                    # token to get the total log-probability.
                    _logprobs += beam_logprobs[beam_idx]

                # Append the results from this beam to the aggregate lists.
                output_ids += _output_ids
                logprobs += _logprobs.tolist()
                beams_done.append(False)

            if all(beams_done):
                # All search beams are done generating text.
                break

            # Keep only the top 'beam_size' beams by total log-probability.
            indices = torch.tensor(logprobs).topk(k=beam_size).indices
            input_ids = [output_ids[idx] for idx in indices]
            beam_logprobs = [logprobs[idx] for idx in indices]

        # Find the predicted beam with highest overall log-probability.
        best_beam_idx = torch.tensor(beam_logprobs).argmax().item()  # type: ignore
        # Decode the predicted token IDs into a text string.
        return self.tokenizer.decode(input_ids[best_beam_idx], skip_special_tokens=True)

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

    language_model_name = 'distilgpt2'
    decoder = GPT2Decoder(language_model_name)
    decoder.load_decoder_checkpoint(cfg.text_decoder_path)
    # model = Decoder.load_from_checkpoint(cfg.text_decoder_path)
    # decoder = DecoderInferenceModel(model=model, tokenizer=get_tokenizer(language_model))

    for i in range(len(text)):
        caption = decoder(text_features[i], 64, 1)
        print(f'result {i}: ')
        print('\t', 'Origin Text:', text[i])
        print('\t', 'Rected Text:', caption)


if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)

