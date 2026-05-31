import torch
import tiktoken
import time
import torch.nn as nn
import pandas as pd
import numpy as np
import urllib.request
import matplotlib
import zipfile
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import custom_flash_attn
from pathlib import Path
from torch.nn import functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from triton_implementation import _flash_attn_forward
from torch.profiler import profile, record_function, ProfilerActivity


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, impl_type):
        self.encoded_text = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(self.encoded_text) - max_length, stride):
            self.input_ids.append(torch.tensor(self.encoded_text[i: i + max_length]))
            self.target_ids.append(torch.tensor(self.encoded_text[i + 1: i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, x):
        return self.input_ids[x], self.target_ids[x]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, impl_type, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "The output dimension must be a multiplicand of num_heads"
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_out = d_out
        self.impl_type = impl_type

        # Sometimes the matrix multiplication requires a bias.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b ,num_tokens ,self.num_heads , self.head_dim).transpose(1,2)
        queries = queries.view(b , num_tokens ,self.num_heads , self.head_dim).transpose(1,2)
        values = values.view(b, num_tokens ,self.num_heads , self.head_dim).transpose(1,2)

        if self.impl_type == "manual":
            attn = queries @ keys.transpose(-2, -1)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn.masked_fill_(mask_bool, -torch.inf)
            attn_weights = torch.softmax(attn / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)

        elif self.impl_type == "functional":
            context_vec = F.scaled_dot_product_attention(
                queries, keys, values, is_causal=True, dropout_p = self.dropout.p if self.training else 0.0
            ).transpose(1,2)

        elif self.impl_type == "triton":
            context_vec = _flash_attn_forward(queries, keys, values)

        elif self.impl_type == "cuda":
            context_vec = custom_flash_attn.forward(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous()
            )

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(config["emb_dim"]))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return norm_x * self.scale + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"])
            )
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            impl_type = cfg["impl_type"],
            qkv_bias = cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.att(x)

        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Utilities/Wrappers
def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset= GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=num_workers)
    return dataloader

def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
   model.eval()
   with torch.no_grad():
       train_loss = calc_loss_loader(train_loader, model, device,
num_batches=eval_iter)
       val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
   model.train()
   return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
   model.eval()
   context_size = model.pos_emb.weight.shape[0]
   encoded = text_to_token_ids(start_context, tokenizer).to(device)
   with torch.no_grad():
       token_ids = generate(
           model=model, idx=encoded,
           max_new_tokens=50, context_size=context_size
       )
       decoded_text = token_ids_to_text(token_ids, tokenizer)
       print(decoded_text.replace("\n", " "))
   model.train()

# OpenAI weights
def assign(left, right):
   if left.shape != right.shape:
       raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
   return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "openai_size": "124M"},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "openai_size": "355M"},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "openai_size": "774M"},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "openai_size": "1558M"},
    }

    implementations = ["cuda", "triton", "manual", "functional"]

    histogram_data = {model: {impl: [] for impl in implementations} for model in model_configs.keys()}

    prompt = (
        "Alpha (originally Alpha AXP) is a 64-bit reduced"
        "instruction set computer (RISC) instruction set"
        " architecture (ISA) developed by Digital Equipment "
        "Corporation (DEC). Alpha was designed to replace 32-bit VAX"
        "complex instruction set computers (CISC) and to be a highly competitive")

    dummy_input = text_to_token_ids(prompt, tokenizer).to(device)

    for impl in implementations:
        print(f"\n{'='*60}")
        print(f"TESTING_IMPLEMENTATION: {impl.upper()}")
        print(f"{'='*60}")

        for model_name, config_updates in model_configs.items():
            print(f"\n-- Loading {model_name} ---")

            cfg = GPT_CONFIG_124M.copy()
            cfg.update({k: v for k, v in config_updates.items() if k != "openai_size"})
            cfg.update({"context_length": 1024, "qkv_bias": True, "impl_type": impl})

            gpt = GPTModel(cfg)
            gpt.eval()

            print(f"Fetching weights for {config_updates['openai_size']}...")
            _, params = download_and_load_gpt2(model_size=config_updates["openai_size"], models_dir="gpt2")
            load_weights_into_gpt(gpt, params)
            gpt.to(device)

            print("Sanity Check Generation:")
            torch.manual_seed(123)
            with torch.no_grad():
                sample_ids = generate(
                    model=gpt, idx=dummy_input, max_new_tokens=20,
                    context_size=cfg["context_length"], top_k=50, temperature=1.0
                )

            print(f"\"{token_ids_to_text(sample_ids, tokenizer).strip()}\"")

            print("Profiling speed...")

            with torch.no_grad():
                for _ in range(10):
                    _ = generate(gpt, dummy_input, max_new_tokens=10, context_size=cfg["context_length"])

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = []

            with torch.no_grad():
                for _ in range(100):
                    starter.record()
                    output = generate(gpt, dummy_input, max_new_tokens=20, context_size=cfg["context_length"])
                    ender.record()
                    torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))

            histogram_data[model_name][impl] = times

            avg_time = np.mean(times)
            std_time = np.std(times)

            print(token_ids_to_text(output, tokenizer))
            print(f">> {model_name} | {impl.upper()} | Time for 20 tokens: {avg_time:.2f} ms ± {std_time:.2f} ms")

            del gpt
            del params
            torch.cuda.empty_cache()

    print("\nSaving individual generation time histograms...")

    color_map = {"triton": "#1f77b4", "functional": "#e377c2", "manual": "#bcbd22", "cuda": "#2ca02c"}

    for model_name in model_configs.keys():
        for impl in implementations:
            data = histogram_data[model_name][impl]

            clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            output_filename = f"histogram_{clean_model_name}_{impl}.png"

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.hist(data, bins=15, color=color_map.get(impl, "#333333"), edgecolor='black', alpha=0.75)
            median = np.median(data)
            ax.axvline(median, color='red', linestyle='dashed', linewidth=1.5, label=f'Med: {median:.1f}ms')

            ax.set_title(f"{model_name}\nAttention Type: {impl.upper()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Latency (ms)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(axis='both', linestyle=':', alpha=0.6)

            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  └─ Saved: {output_filename}")

    print("\nAll individual histograms generated successfully.")
