# A little Language Model

A 39M (lil) parameter model trained on ~8B tokens, on 2xA100 for approximately 2 hours. More details below.

## Introduction

> What I cannot create, I do not understand - Richard Feynman

Simply understanding the model architecture is not enough to fully grasp how these models are trained. This project is the outcome of this realization and the frustration on how abstractions limit our learning process (eg. huggingface transformers) at least when we are starting out. The best thing to do is to implement everything from scratch, within minimal abstraction. Well, this is what this project does. With this project, I plan to add everything(code + my notes) from training tokenizers to the post-tranining phases. One may consider it as a roadmap, but it might not be enough and at the end you will have your own roadmap, so just consider it as an outline or introduction to training Large Language Models.

## Prerequisite

You should have basic understanding of how transformer model works. A great way to start is by watching and implementing yourself [Karpathy's zero to hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series til part 5. Afterwards, you can take a look at Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), and then visit Karpathy's [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=ZyI_mMpGKGfUlkFV). This is just my recommendation, please make sure to visit them in any order as per your need.

## lilLM Architecture

The architecture differs from transformers architecture in that it uses.

- RMSNorm instead of LayerNorm
- Rotary Positional Embedding instead of Absolute Positional Embedding
- SwiGLU activations instead of ReLU
- Grouped Query Attention instead of Multi-head Attention

Finally, the architecture becomes similar to what is used in Llama 3 models.

![architecture](/misc/lilLM_architecture.png)

| Attribute     | `vocab_size` | `d_model` | `n_layers` | `max_seq_len` | `q_heads` | `kv_heads` | `max_batch_size` |
| ------------- | ------------ | --------- | ---------- | ------------- | --------- | ---------- | ---------------- |
| Default Value | `2**13`      | `512`     | `12`       | `512`         | `16`      | `8`        | `32`             |

### Tokenizer

This is the first step in training LM. As LMs can't take text as an input we need to convert text to numbers. We build our own vocabulary to map tokens to numbers. A great way understand the whole concept is to watch karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=3301s). You might need some knowledge about unicode and utf-8 to completely grasp the concept in detail for which you can look at my notes on [Tokenizers](https://cohlem.github.io/sub-notes/tokenization/). In this project, We train huggingface tokenizer (this is the only abstraction that we use) to train our tokenizer. It was trained on 0.1% of [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext). Recommended way would be to train the tokenizer on diverse and large dataset to get the best compression rate. For simplicity, I wanted my model to just be able to converse well, I opted for this small subset of the dataset which you can find [here](https://huggingface.co/datasets/CohleM/openweb-800k)

### Model architecture

As described above, the architecture deviates from original transformer model. A couple of changes are:

#### RMSNorm

Please read this paper [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467), A simple conclusion from the paper is that we don't need to calculate the mean across layers while performing normalization as we do in Layer Normalization, just maintaining the variation((scaling)) is sufficient.

#### Rotary Positional Embedding

Instead of adding extra positional embedding to our token embeddings, we simply rotate our token embeddings. I would first recommend watching this video [RoPE (Rotary positional embeddings) explained](https://www.youtube.com/watch?v=GQPOtyITy54), then read the paper [ROFORMER](https://arxiv.org/pdf/2104.09864) and finally look at my notes on [RoPE](https://cohlem.github.io/sub-notes/rope/) where I explain ROPE with respect to the code that we use in this project.

#### SwiGLU activations

Take a look at this simple and straightforward blog on [SwiGLU: GLU Variants Improve Transformer (2020)](https://kikaben.com/swiglu-2020/)

#### Grouped Query Attention

Instead of using multiple heads in our attention, we simply divide K and V to groups and repeat those K,V to q_heas/kv_heads times, and then perform attention. Why? since K and V are repeated, the data movement within GPU is minimized cause it is the most expensive task and is a bottleneck to our training. To understand better, take a look at this video [Variants of Multi-head attention](https://www.youtube.com/watch?v=pVP0bu8QA2w) and then read my notes on [Grouped Query Attention](https://cohlem.github.io/sub-notes/kv-cache-gqa/)

### Training Details

#### Pretraining Data

The model was trained [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext), which is close to 10 billion tokens according to our tokenizer, but the model was only trained on ~8B tokens (credits ran out :( ).

#### Compute

It was trained on 2XA100 for approximately 2.5 hours.

This is the specification of machine that I used. GPU was rented from [Tensordock](https://www.tensordock.com)

| **Category**      | **Details**            |
| ----------------- | ---------------------- |
| **Storage**       | 300 GB                 |
| **vCPUs**         | 40 AMD EPYC 7513 vCPUs |
| **RAM**           | 80 GB RAM              |
| **GPU**           | 2x A100 SXM4 80 GB     |
| **Compute Price** | $2.400000/hour         |
| **Storage Price** | $0.075000/hour         |
| **Total Price**   | $2.475000/hour         |

#### Train/Val loss

![loss](misc/loss.png)

## Sample outputs after pretraining

Input

```
Bernie Sanders of Vermont would seek a recount. The delay postponed a definitive answer to whether Clinton had made a clean sweep of five big primaries on
```

Output

```
Bernie Sanders of Vermont would seek a recount. The delay postponed a definitive answer to whether Clinton had made a clean sweep of five big primaries on opening day of the Democratic nominating process.\n\nIn response, Sanders theorized that it was possible for her to then-choice’s hand to escalate into a “unprecedented vote” to take the nomination. However, his standing to refrain from carrying coal companies in the wilderness at the time of her Nov. 8 pick-ing defeat surprised the race.\n\nTrump said Wednesday morning that he will back Clinton to replace the incumbent U.S. senator who ran against Bernie Sanders on Tuesday, in a 33-16 historic win over Hillary Clinton. Though, given that both Hillary Clinton and Bernie Sanders enjoyed a restricted number of fallen out of the race on the set of their Iowa primary
```

Input

```
The latest Marvel movie has shattered box office records, grossing over $1 billion worldwide in just two weeks. Fans have praised the
```

Output

```
The latest Marvel movie has shattered box office records, grossing over $1 billion worldwide in just two weeks. Fans have praised the icons like Roc Nation and Meet The Press for the budgetary ramifications Ain’t Not a breakout promotion.\n\nIn the second week of December, Marvel announced Monday that various Marvel games and Daredevil: The Desolation of holding off it would leave Friday to Evil Geniuses. The Daredevil announced Monday that The Clone Wars is now open and ready for release in late June.
```

## Quick Start

### Training from Scratch

#### Install dependencies

```
git clone https://github.com/CohleM/lilLM.git
```

```
pip install -r requirements.txt
```

#### Train Tokenizer

I plan to make this more startforward by adding commandline arguments, but for now please follow the steps described

Download the data from [here](https://huggingface.co/datasets/CohleM/openweb-800k) and convert it to jsonl format, open the `train_custom_tokenizer.py` file and replace the file_path with your path/to/your_jsonl_file and then

```python
python train_custom_tokenizer.py
```

Tokenizer will be stored in `/model/tokenizer`.

#### Download and Tokenize pretraining data

```
python data/pretraining/process.py
```

It will download the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset from huggingface and tokenize the whole dataset using our tokenizer saved in `/model/tokenizer` and save tokenized files as `train.bin` and `val.bin`.These are the binary files for our tokenized dataset. `train.bin` results in ~20GB. The reason for tokenizing it beforehand is because we want to maximize our GPU utilization. Since tokenization is a CPU bound task, we can do it before hand while allowing our GPU train more tokens during training.

#### Pretrain

If you have Nx GPU per node run.

```
torchrun --standalone --nproc_per_node=2 pretrain.py
```

If you only have one GPU run,

```
python pretrain.py
```

Please also take a look at default config parameters in `model/config.py` and in `pretrain.py`

## TODO

### Post Training Stages

- Finetune using SFT and DPO

### Architectural Changes

- Add Mixture of Experts (MoE)

### Inference

- Add Inference file
