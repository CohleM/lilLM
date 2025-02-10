import os
import random
import time
import math

import wandb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext


from model.config import Config
from model.model import LilLM

cur_path = os.path.dirname(os.path.abspath(__file__))

print('gg', cur_path)
data_path = os.path.join(cur_path, 'data/pretraining')
out_dir = ''

batch_size = 64
block_size = 512
lr = 1e-6
max_iters = 20000
grad_clip = 1.0
eval_interval = 200 # do eval every 200 interval
eval_iters = 20 # for accumulate eval losses for 200 iters
best_val_loss = 1e9


# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
learning_rate = 6e-4 # max learning rate


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# we want to do gradient update per 0.5M tokens, but our GPU can't fit that size i.e lets say our we have block_size=1024, it would take
# 488 batch_size to do it in the single run, but our gpu can't fit it, so we divide further, i.e we accumulate gradient on smaller batch, once we
# have accumulated gradients for 0.5M tokens, we do the update, otherwise just accumulate the gradients. keeping batch_size=16, block_size=1024, we need
# divide into 0.5*1e6/(16*1024) steps, which we name gradient_accumulation_steps
gradient_accumulation_steps = 16 
init_from = 'scratch'


# wandb logging
wandb_project = 'LilLM'
wandb_run_name = 'GPU_RUN'


compile = True
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
  init_process_group(backend="nccl")
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
  seed_offset = ddp_rank # each process gets a different seed
  assert gradient_accumulation_steps % ddp_world_size == 0
  gradient_accumulation_steps //= ddp_world_size

else:
      # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

print('tokens per iteration', gradient_accumulation_steps * ddp_world_size * batch_size * block_size)
torch.manual_seed(1337 + seed_offset) # set different seed for differnt gpus

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



@torch.no_grad()
def estimate_losses():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    each_loss = 0
    for i in range(eval_iters):
      x,y = data_loader(data_path, split)
      with ctx:
        logits,loss = model(x,targets=y)
      each_loss += loss.item()
    out[split] = each_loss/eval_iters
  model.train()
  return out


# Data Loader
def data_loader(data_path, split):
    filename = os.path.join(data_path, f'{split}.bin')
    data = np.memmap(filename, mode='r', dtype=np.uint16) # please make sure to load the bin file with correct dtype, this costed be some $$
    ids = torch.randint((len(data)-block_size), (batch_size,))

    X = torch.stack([ torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ids])
    Y = torch.stack([ torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ids])

    if device_type=='cuda':
        # pin to (page-locked) memory in the host (CPU) memory, so the write to GPU is faster, also enable async data transfer(non_blocking=True)
        return X.pin_memory().to(device,non_blocking=True), Y.pin_memory().to(device,non_blocking=True)
    else:
        return X.to(device), Y.to(device)



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn (eg. convolutions nn.Conv)
device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# int model parameters
model_config = Config(max_seq_len = block_size, max_batch_size = batch_size)


if init_from == 'scratch':
  model = LilLM(model_config)
elif init_from == 'resume': # resume from a checkpoint
  pass


if wandb and master_process:
  wandb.init(project=wandb_project, name=wandb_run_name, config=model_config)

model.to(device)
# scaler is required if we use fp16, cause gradients and loss need to be scaled because of lower range i.e 5-bit exponent (gradients, may explode or vanish)
# not needed for bf16, cause it has 8-bit exponent
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

if compile and ddp:
  print('torch compiling the model..')
  unoptimized_model = model
  model = torch.compile(model)

# wrap model in ddp container

if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])

num_iter = 0
raw_model = model.module if ddp else model

while True:
  # pick learning rate
  lr = get_lr(num_iter) if decay_lr else learning_rate

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  # evaluation and log losses to wandb
  if num_iter % eval_interval == 0 and master_process:
    losses = estimate_losses()
    print(f'Steps {num_iter} train loss:{losses["train"]} val loss: {losses["val"]} ')
    if wandb:
      wandb.log({
        'iter': num_iter,
        'train_loss': losses['train'],
        'val_loss': losses['val'],
        'lr': lr
      })

    # Save checkpoint with best loss
    if losses['val'] < best_val_loss:
      best_val_loss = losses['val']
      
      checkpoint = {
        'model' : raw_model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_val_loss' : best_val_loss,
        'num_iter' : num_iter,
        'config' : model_config
      }
      torch.save(checkpoint, os.path.join(out_dir, 'best_model.pt'))

  if (master_process and (num_iter % 250 == 0 or max_iters-1)) :
    sample_prompt = 'Hi! I am a Language Model, and '
    sample_token  = torch.tensor(tokenizer.encode(sample_prompt)).unsqueeze(dim=0).to(device)
    for i in range(4):
      # [[2]] is the ending token 
      out = model.generate(sample_token, torch.tensor([[2]]).to(device))
      print(f' {i} output: {tokenizer.decode(out.squeeze())}')


  for micro_step in range(gradient_accumulation_steps):
    t1 = time.time()
    x,y = data_loader(data_path, 'train')
    # x,y = torch.randint(0,10,(10,256)).to(device), torch.randint(0,10,(10,256)).to(device)

    if ddp:
      # in DDP training we only need to sync gradients at the last micro step.
      # the official way to do this is with model.no_sync() context manager, but
      # I really dislike that this bloats the code and forces us to repeat code
      # looking at the source of that context manager, it just toggles this variable
      model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
      logits,loss = model(x,targets=y)
      loss = loss / gradient_accumulation_steps
    # gradient sync happens here
    # why scale??
    # fp16 can only store 5-bit exponents, to preserve large numbers from getting zerod
    # the scaler multiplies our numbers with scaler eg 1024
    # exp: np.float16(0.0000000123423543) will result in 0, cause there are not enough
    # exponent to store this number, so when we do this np.float16(0.0000000123423543*1024)
    # result is np.float16(1.264e-05), its a scaled value.
    scaler.scale(loss).backward()

  if grad_clip != 0.0:
    # unscale the gradients, cause we need higher precision in AdamW optimzers, and we don't use mixed-precision
    scaler.unscale_(optimizer)
    # clip the gradients to prevent vanishing gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

  scaler.step(optimizer)
  # adjusts the loss scaling factor dynamically, eg, if prev step caused overflow
  # decrease scaling factor, else increase scaling factor
  scaler.update()
  optimizer.zero_grad(set_to_none=True)
  num_iter +=1
  t2 = time.time()
  print(f'Iteration: {num_iter} loss: {(loss.item() * gradient_accumulation_steps):.4f} time_taken: {(t2-t1):.2f}' )
  if num_iter > max_iters:
    break

if ddp:
    destroy_process_group()

