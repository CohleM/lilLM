import json
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import os 
import random

random.seed(42)
# Define the data loader function separately
def read_texts_from_jsonl(file_path):
    """Reads text data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']



def train_tokenizer(file_path):

    tokenizer = Tokenizer(models.BPE()) 
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # convert character into bytes, and don't add space to the beginning of text

    # tokens: <unk> - for token it hasn't seen during training,
    # <s> - start of sentence
    # </s> - end of sentence
    special_tokens = ["<unk>", "<s>", "</s>"]
    
    # set configs for BPE trainer, 
    trainer = trainers.BpeTrainer(
        vocab_size=8192, # 2^13
        special_tokens=special_tokens,  
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.decoder = decoders.ByteLevel()

    # Read dataset from jsonl 
    texts = read_texts_from_jsonl(file_path)
    
    #train using iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer_dir = "./model/tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/tokenizer")
    
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False
#         "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }
    
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")
    


if __name__ =='__main__':
    file_path = 'openwebtext_800k.jsonl'
    train_tokenizer(file_path)
