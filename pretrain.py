import random
import os
import time
import math
import argparse

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
from model.utils import calculate_transformer_flops

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/pretraining")
DEFAULT_OUT_DIR = ""
DEFAULT_BATCH_SIZE = 128
DEFAULT_BLOCK_SIZE = 512
DEFAULT_MAX_ITERS = 20000
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EVAL_INTERVAL = 200  # do eval every 200 interval
DEFAULT_LOG_INTERVAL = 10
DEFAULT_EVAL_ITERS = 20  # for accumulate eval losses for 200 iters
DEFAULT_BEST_VAL_LOSS = 1e9
# learning rate decay settings
DEFAULT_DECAY_LR = True  # whether to decay the learning rate
DEFAULT_WARMUP_ITERS = 2000  # how many steps to warm up for
DEFAULT_LR_DECAY_ITERS = 600000  # should be ~= max_iters per Chinchilla
DEFAULT_MIN_LR = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
DEFAULT_LEARNING_RATE = 6e-4  # max learning rate
DEFAULT_RUNNING_MFU = -1
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
DEFAULT_DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# we want to do gradient update per 0.5M tokens, but our GPU can't fit that size i.e lets say our we have block_size=1024, it would take
# 488 batch_size to do it in the single run, but our gpu can't fit it, so we divide further, i.e we accumulate gradient on smaller batch, once we
# have accumulated gradients for 0.5M tokens, we do the update, otherwise just accumulate the gradients. keeping batch_size=16, block_size=1024, we need
# divide into 0.5*1e6/(16*1024) steps, which we name gradient_accumulation_steps
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
#init_from = "scratch"
DEFAULT_INIT_FROM = "scratch"
# wandb logging
DEFAULT_WANDB_PROJECT = "LilLM"
DEFAULT_WANDB_RUN_NAME = "GPU_RUN_NEW"

DEFAULT_COMPILE = True


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters ):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_losses(model, data_path, batch_size, block_size, device, eval_iters, device_type, ctx):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        each_loss = 0
        for i in range(eval_iters):
            x, y = data_loader(data_path, split,batch_size, block_size, device, device_type)
            with ctx:
                logits, loss = model(x, targets=y)
            each_loss += loss.item()
        out[split] = each_loss / eval_iters
    model.train()
    return out


# Data Loader
def data_loader(data_path, split, batch_size, block_size, device, device_type):
    filename = os.path.join(data_path, f"{split}.bin")
    data = np.memmap(
        filename, mode="r", dtype=np.uint16
    )  # please make sure to load the bin file with correct dtype, this costed be some $$
    ids = torch.randint((len(data) - block_size), (batch_size,))

    X = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ids]
    )
    Y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ids
        ]
    )

    if device_type == "cuda":
        # pin to (page-locked) memory in the host (CPU) memory, so the write to GPU is faster, also enable async data transfer(non_blocking=True)
        return X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        return X.to(device), Y.to(device)


def set_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(f"cuda:{ddp_local_rank}")
        return ddp, ddp_rank, ddp_local_rank, ddp_world_size
    return 0, 0, 0, 1


def main(args):
    ddp, ddp_rank, ddp_local_rank,ddp_world_size = set_distributed()
    #device = f"cuda:{ddp_local_rank}" if ddp else 'cpu'
    if ddp:
        device = f"cuda:{ddp_local_rank}"
    else:
        device = args.device

    master_process = ddp_rank == 0
    torch.manual_seed(1337 + ddp_rank)  # set different seed for differnt gpus
    assert args.gradient_accumulation_steps % ddp_world_size == 0
    args.gradient_accumulation_steps //= ddp_world_size

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn (eg. convolutions nn.Conv)
    device_type = "cuda" if "cuda" in device else "cpu"
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    print("tokens per iteration", args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size)
    model_config = Config(max_seq_len=args.block_size, max_batch_size=args.batch_size)
    # Flops estimation
    flops_per_model = calculate_transformer_flops(
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        key_size=model_config.d_model/model_config.q_heads,
        num_heads=model_config.q_heads,
        ffw_size=model_config.hidden_dim if model_config.hidden_dim is not None else 4*model_config.d_model,
        num_layers=model_config.n_layers
    )
    flops_per_step = flops_per_model * model_config.max_batch_size * args.gradient_accumulation_steps * ddp_world_size
    running_mfu = -1.0
    # Train from scratch or from a checkpoint
    if args.init_from == "scratch":
        model = LilLM(model_config)
        num_iter = 0
        best_val_loss = DEFAULT_BEST_VAL_LOSS
    elif args.init_from == "resume":  # resume from a checkpoint
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(os.path.join(args.out_dir, 'best_model.pt'), map_location=device)
        model_config = checkpoint['config'] 
        model = LilLM(model_config)
        # saved model keys contain some prefix, we need to rename them to our original name
        state_dict = checkpoint['model'] 
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict) 
        num_iter = checkpoint['num_iter']
        best_val_loss = checkpoint['best_val_loss']

    if args.init_from == "resume" and master_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=checkpoint.get('wandb_run_id'),  # You'll need to save this in checkpoint
            resume="must"
        )
    elif wandb and master_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=model_config)

    model.to(device)
    # scaler is required if we use fp16, cause gradients and loss need to be scaled because of lower range i.e 5-bit exponent (gradients, may explode or vanish)
    # not needed for bf16, cause it has 8-bit exponent
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


    if DEFAULT_COMPILE and ddp:
        print("torch compiling the model..")
        unoptimized_model = model
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model


    while True:
        # pick learning rate
        lr = get_lr(num_iter, args.learning_rate, args.min_lr, args.warmup_iters, args.lr_decay_iters) if DEFAULT_DECAY_LR else args.learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluation and log losses to wandb
        if num_iter % args.eval_interval == 0 and master_process:
            losses = estimate_losses(model, args.data_path, args.batch_size, args.block_size, device, args.eval_iters, device_type, ctx)
            print(
                f'Steps {num_iter} train loss:{losses["train"]} val loss: {losses["val"]} '
            )
            if wandb:
                wandb.log(
                    {
                        "iter": num_iter,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                    }
                )

            # Save checkpoint with best loss
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "num_iter": num_iter,
                    "config": model_config,
                    "wandb_run_id": wandb.run.id
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pt"))

        t1 = time.time()
        for micro_step in range(args.gradient_accumulation_steps):
            x, y = data_loader(args.data_path, "train",args.batch_size, args.block_size,device, device_type)
            # x,y = torch.randint(0,10,(10,256)).to(device), torch.randint(0,10,(10,256)).to(device)

            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(x, targets=y)
                loss = loss / args.gradient_accumulation_steps
            # gradient sync happens here
            # why scale??
            # fp16 can only store 5-bit exponents, to preserve large numbers from getting zerod
            # the scaler multiplies our numbers with scaler eg 1024
            # exp: np.float16(0.0000000123423543) will result in 0, cause there are not enough
            # exponent to store this number, so when we do this np.float16(0.0000000123423543*1024)
            # result is np.float16(1.264e-05), its a scaled value.
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            # unscale the gradients, cause we need higher precision in AdamW optimzers, and we don't use mixed-precision
            scaler.unscale_(optimizer)
            # clip the gradients to prevent vanishing gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        # adjusts the loss scaling factor dynamically, eg, if prev step caused overflow
        # decrease scaling factor, else increase scaling factor
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iter += 1
        t2 = time.time()
        dt = t2 - t1
        if num_iter % args.log_interval == 0 and master_process:
            if num_iter >=5:
                flops_promised = 312e12 # flops that we can do in A100 for bfloat16
                mfu = flops_per_step / (flops_promised * (dt))
                running_mfu =  mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(
                    f"iteration: {num_iter} loss: {(loss.item() * args.gradient_accumulation_steps):.4f} time_taken: {(dt):.2f}, mfu : {mfu*100:.2f}%"
                )
        if num_iter > args.max_iters:
            break

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LilLM')
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="Block size for training.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Maximum learning rate.")
    parser.add_argument("--min_lr", type=float, default=DEFAULT_MIN_LR, help="Minimum learning rate.")
    parser.add_argument("--max_iters", type=int, default=DEFAULT_MAX_ITERS, help="Maximum number of iterations.")
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP, help="Gradient clipping value.")
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL, help="Evaluation interval.")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL, help="Logging interval.")
    parser.add_argument("--eval_iters", type=int, default=DEFAULT_EVAL_ITERS, help="Number of iterations for evaluation.")
    parser.add_argument("--warmup_iters", type=int, default=DEFAULT_WARMUP_ITERS, help="Number of warmup iterations.")
    parser.add_argument("--lr_decay_iters", type=int, default=DEFAULT_LR_DECAY_ITERS, help="Number of iterations for learning rate decay.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use for training (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="Data type for training (e.g., 'float16', 'bfloat16').")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT, help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=DEFAULT_WANDB_RUN_NAME, help="Wandb run name.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to save checkpoints.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to the training data.")
    parser.add_argument("--init_from", type=str, default=DEFAULT_INIT_FROM, help="resume or scratch")
    args = parser.parse_args()
    main(args)

