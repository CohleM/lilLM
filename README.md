A little Language Model


### Tokenizer
Trained on 0.1% of [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext).

`vocab_size`: 2***^13

### Pretraining

Download and tokenize the dataset and save it to .bin files. 
*****Key Advantages**
- reduces tokenization overhead, cause data is already tokenized.
- saves storage space.


```python
from transformers import GPT2Tokenizer
import numpy as np

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example text
text = "Hello, world! This is an example of tokenization."

# Tokenize the text
tokens = tokenizer.encode(text)

# Save raw text to a file
with open("raw_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Save tokenized data to a binary file
tokens_np = np.array(tokens, dtype=np.uint16)
tokens_np.tofile("tokenized_data.bin")

# Compare file sizes
import os
raw_text_size = os.path.getsize("raw_text.txt")
tokenized_size = os.path.getsize("tokenized_data.bin")

print(f"Raw text size: {raw_text_size} bytes")
print(f"Tokenized size: {tokenized_size} bytes")
```

output
```
Raw text size: 49 bytes
Tokenized size: 24 bytes
```

*****Explanation**
`len(list(text.encode('utf-8')))` counts to 49, and each decimal takes 1 byte.

using our tokenizer we compress the tokens within our vocab range (2^13) and save each
token in uint16 (cause 2^13 < 2^16, very much possible). and each token takes 2 bytes.

in our case `len(tokens_np)*2` is 12 which is 12x2 bytes.







