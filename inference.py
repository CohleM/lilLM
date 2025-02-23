import time
import torch
import argparse
import os

from transformers import AutoTokenizer

from model.model import LilLM
from model.config import Config

DEFAULT_TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/tokenizer')
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_790_sft.pt')
DEFAULT_TEXT = 'What is the capital of France?'
DEFAULT_MODEL_TYPE = 'sft'

def add_chat_format(text):
    template = f"<r0>user<r1>" + f"{text}</r2><r0>assistant<r1>"
    return template


if __name__=='__main__':
    torch.serialization.add_safe_globals([Config])
    parser = argparse.ArgumentParser(description='Sample text')
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Input to the model")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, help="sft model or pretrained")
    
    

    args = parser.parse_args()

    #model_path = 'best_model_790_sft.pt' 
    model = LilLM(Config(flash=False))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    #checkpoint = torch.load('/Users/cohlem/Projects/Experimentation/lillm/best_model_790_sft.pt', map_location=device)
    checkpoint = torch.load(args.model_path, map_location=device)


    state_dict = checkpoint['model'] 
    unwanted_prefix = '_orig_mod.'

    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(checkpoint['model'])

    template_text = add_chat_format(args.text) if args.model_type == "sft" else args.text
    
    t0 = time.time()
    start_prompt = torch.tensor(tokenizer.encode(template_text)).unsqueeze(dim=0).to(device)
    eos = torch.tensor([[2]]).to(device)
    print(tokenizer.decode(model.generate(start_prompt, eos).squeeze()))
    t1 = time.time()

    print(f'\n Completed in {t1-t0} seconds')

