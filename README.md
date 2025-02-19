# A little Language Model

A 39M (lil) parameter model trained on ~8B tokens, on 2xA100 for approximately 2 hours. More details below.

## Introduction

> What I cannot create, I do not understand - Richard Feynman

Simply understanding the model architecture is not enough to fully grasp how these models are trained. This project is the outcome of this realization and the frustration on how abstractions limit our learning process (eg. huggingface transformers) at least when we are starting out. The best thing to do is to implement everything from scratch, within minimal abstraction. Well, this is what this project does. With this project, I plan to add everything(code + my notes) from training tokenizers to the post-tranining phases. One may consider it as a roadmap, but it might not be enough and at the end you will have your own roadmap, so just consider it as an outline or introduction to training Large Language Models.

## Prerequisite

You should have basic understanding of how transformer model works. A great way to start is by watching and implementing yourself [Karpathy's zero to hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series til part 5. Afterwards, you can take a look at Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), and then visit Karpathy's [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=ZyI_mMpGKGfUlkFV). This is just my recommendation, please make sure to visit them in any order as per your need.

## Architecture

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
