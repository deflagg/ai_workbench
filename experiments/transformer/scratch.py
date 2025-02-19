#!/usr/bin/env python
# my_script.py

import os
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# For data loading (example: wikitext)
from datasets import load_dataset

# For tokenization and GPT-2 embeddings
from transformers import AutoTokenizer, GPT2Model

# For logging & checkpointing
import wandb

# For visualization
from torchviz import make_dot

from tqdm import tqdm

###############################################################################
# 1) DATA LOADING / TOKENIZATION
###############################################################################

def load_wikitext2():
    """
    Example function for reading WikiText-2 from Hugging Face
    and returning lists of raw text lines.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
    val_texts   = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]
    return train_texts, val_texts


def load_wikitext103():
    """
    Example function for reading WikiText-103 from Hugging Face
    and returning lists of raw text lines.
    """
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
    val_texts   = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]
    return train_texts, val_texts


class CustomTextDataset(Dataset):
    """
    A simple Dataset for tokenized text data (list of list of token IDs).
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def prepare_data(
    load_dataset_fn,
    tokenizer,
    batch_size=8,
    seq_len=64,
    shuffle_train=True,
    shuffle_val=False
):
    """
    Generic data preparation function.

    Parameters
    ----------
    load_dataset_fn : callable
        A user-provided function that loads raw data and returns (train_texts, val_texts).

    tokenizer : PreTrainedTokenizer (or any callable)
        A tokenizer or encoding function that converts text -> list of token IDs.

    batch_size : int
        Batch size for the DataLoader.

    seq_len : int
        Maximum sequence length (for truncation/padding).

    shuffle_train : bool
        Whether to shuffle the training DataLoader.

    shuffle_val : bool
        Whether to shuffle the validation DataLoader.

    Returns
    -------
    train_loader, val_loader, tokenizer
    """
    # Create cache filename based on parameters
    cache_file = f"tokenized_cache_{load_dataset_fn.__name__}_{seq_len}.pkl"

    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading tokenized data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            train_encodings, val_encodings = pickle.load(f)
    else:
        print("Processing dataset from scratch...")
        train_texts, val_texts = load_dataset_fn()

        def tokenize(lines):
            return tokenizer(
                lines,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"].tolist()

        train_encodings = tokenize(train_texts)
        val_encodings   = tokenize(val_texts)

        # Save to cache
        print(f"Saving tokenized data to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump((train_encodings, val_encodings), f)

    # 3) Wrap them in a Dataset & DataLoader
    train_dataset = CustomTextDataset(train_encodings)
    val_dataset   = CustomTextDataset(val_encodings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=shuffle_val)

    return train_loader, val_loader, tokenizer

###############################################################################
# 2) PRETRAINED GPT-2 EMBEDDING
###############################################################################

class GPT2Embedding(nn.Module):
    """
    Wrapper that loads GPT-2 and extracts only the input embeddings
    (token + positional embeddings), frozen. This means we do NOT train
    or finetune these embeddings – we only read them.
    """
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        # Freeze all GPT-2 params
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        """
        Return just the input (token+position) embeddings,
        NOT the hidden states after transformer blocks.
        """
        # Forward with output_hidden_states=True
        outputs = self.gpt2(input_ids, output_hidden_states=True)
        # The first hidden_states (index 0) is the embedding output
        # (token embeddings + positional embeddings).
        embeddings = outputs.hidden_states[0]  # shape: (B, S, hidden_size)
        return embeddings

###############################################################################
# 3) MODEL DEFINITION (GENERIC TRANSFORMER) but uses GPT-2 embeddings
###############################################################################

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, S, d_model)
        ln1 = self.layernorm1(x)
        attn_out, _ = self.self_attn(ln1, ln1, ln1)
        x = x + self.dropout(attn_out)

        ln2 = self.layernorm2(x)
        ff_out = self.feed_forward(ln2)
        x = x + self.dropout(ff_out)
        return x


class GenericTransformer(nn.Module):
    """
    This model uses the pretrained GPT-2 embedding for input_ids,
    and then a stack of custom Transformer blocks on top.
    """
    def __init__(
        self,
        vocab_size,            # Usually from GPT-2 tokenizer (e.g. 50257)
        n_heads=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        gpt2_model_name="gpt2"
    ):
        super().__init__()

        # GPT-2 base hidden dimension is 768 for 'gpt2'
        # (If using 'gpt2-medium', etc., the hidden_size may differ.)
        self.embedding_module = GPT2Embedding(model_name=gpt2_model_name)
        d_model = self.embedding_module.gpt2.config.hidden_size

        # Build custom Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.layernorm_out = nn.LayerNorm(d_model)
        self.output_fc     = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, S) token IDs
        # 1) get GPT-2's pretrained token+pos embeddings
        with torch.no_grad():
            # We only freeze GPT-2 part
            emb = self.embedding_module(x)  # shape (B, S, d_model)

        # 2) pass embeddings through our *trainable* Transformer blocks
        hidden = emb
        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.layernorm_out(hidden)
        logits = self.output_fc(hidden)  # shape: (B, S, vocab_size)
        return logits

###############################################################################
# 4) EVALUATION & INFERENCE
###############################################################################

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate cross-entropy on a (sub)set of data.
    For language modeling, we usually predict the next token, 
    so we shift inputs by 1 position.
    """
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            B, S, V = logits.shape
            if S < 2:
                continue
            # Next-token prediction
            loss_ce = criterion(
                logits[:, :-1, :].reshape(B*(S-1), V),
                batch[:, 1:].reshape(-1)
            )
            total_loss += loss_ce.item()
            count += 1
    return total_loss / max(count, 1)


def sample_inference(model, tokenizer, prompt="Hello", max_new_tokens=20, device="cpu", temperature=0.8):
    """
    Autoregressive sampling with temperature.
    Higher temperature = more random/creative
    Lower temperature = more focused/deterministic
    """
    model.eval()
    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        # logits shape: (B=1, S, vocab_size)
        next_token_logits = logits[:, -1, :]       # (1, vocab_size)

        # Apply temperature
        scaled_logits = next_token_logits / temperature
        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        # Sample from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


###############################################################################
# 5) TRAINING LOOP WITH CHECKPOINTING & MID-EPOCH VALIDATION
###############################################################################

def train_model(
    model,
    train_loader,
    val_loader,
    tokenizer=None,   # optional, for sampling
    epochs=1,
    lr=1e-4,
    device="cpu",
    disable_wandb=False,
    wandb_project="gpt2-embed-training",
    load_checkpoint=None,
    save_checkpoint=None,
    checkpoint_every=None,
    sample_prompt="Hello",
    validate_after=None,
    partial_val_loader=None
):
    """
    Generic training function for any model on any text dataset.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.

    train_loader : DataLoader
        Training data loader.

    val_loader : DataLoader
        Full validation data loader (used at the end of each epoch).

    tokenizer : (optional)
        Tokenizer for sampling (if model is a text generative model).

    epochs : int
        Number of epochs.

    lr : float
        Learning rate.

    device : torch.device or str
        "cpu" or "cuda".

    disable_wandb : bool
        If True, wandb logging is disabled.

    wandb_project : str
        Name of the Weights & Biases project.

    load_checkpoint : str or None
        Path to checkpoint file to resume from (if present).

    save_checkpoint : str or None
        Path to save final (or periodic) checkpoints.

    checkpoint_every : int or None
        If set, save a checkpoint every N steps.

    sample_prompt : str
        Text prompt for sampling demonstration after each epoch (optional).

    validate_after : int or None
        If set, run a partial validation every N steps within an epoch.

    partial_val_loader : DataLoader or None
        Subset of the validation loader used for mid-epoch validation.

    Returns
    -------
    model, tokenizer
    """
    # 1) Initialize W&B
    if disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    optimizer = torch.optim.AdamW(
        # Only optimize our custom transformer parameters, not GPT-2 embeddings
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    wandb.watch(model, log_freq=100)  # optional: log gradients/params

    start_epoch = 0
    global_step = 0

    # If a checkpoint is provided, load it and resume training
    if load_checkpoint is not None and os.path.isfile(load_checkpoint):
        print(f"Loading state from checkpoint: {load_checkpoint}")
        ckpt = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"Resumed training from epoch={start_epoch}, global_step={global_step}")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                batch = batch.to(device)

                logits = model(batch)
                B, S, V = logits.shape
                if S < 2:
                    # skip short sequences
                    continue

                loss_ce = criterion(
                    logits[:, :-1, :].reshape(B*(S-1), V),
                    batch[:, 1:].reshape(-1)
                )
                loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Logging
                if global_step % 50 == 0:
                    print(f"Epoch {epoch}, Step {global_step} | CE: {loss_ce.item():.4f}")
                    wandb.log({
                        "train/loss": loss_ce.item(),
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }, step=global_step)

                # MID-EPOCH VALIDATION ON A SUBSET
                if validate_after is not None and partial_val_loader is not None:
                    if (global_step % validate_after) == 0 and global_step > 0:
                        val_subset_loss = evaluate(model, partial_val_loader, criterion, device)
                        print(f"Step {global_step} | Partial Val CE: {val_subset_loss:.4f}")
                        wandb.log({"val/partial_loss": val_subset_loss}, step=global_step)

                # Save partial checkpoint if needed
                if checkpoint_every and global_step % checkpoint_every == 0 and global_step > 0:
                    if save_checkpoint is not None:
                        save_dict = {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step
                        }
                        torch.save(save_dict, save_checkpoint)
                        print(f"Saved checkpoint at step={global_step} to {save_checkpoint}")

                        # Optionally log checkpoint to W&B
                        artifact = wandb.Artifact(
                            name=f"model-checkpoint-{wandb.run.id}",
                            type="model",
                            description=f"Checkpoint at step {global_step}"
                        )
                        artifact.add_file(save_checkpoint)
                        wandb.log_artifact(artifact)

                global_step += 1

            # End of epoch metrics
            epoch_loss_avg = epoch_loss / max(num_batches, 1)
            val_loss       = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch} | Train CE: {epoch_loss_avg:.4f} | Val CE: {val_loss:.4f}")
            wandb.log({
                "train/epoch_loss": epoch_loss_avg,
                "val/loss": val_loss,
                "epoch": epoch
            }, step=global_step)

            # Optionally sample at the end of epoch
            if tokenizer is not None and sample_prompt:
                gen_text = sample_inference(
                    model,
                    tokenizer,
                    prompt=sample_prompt,
                    max_new_tokens=20,
                    device=device
                )
                print("[Sample inference]", gen_text)
                wandb.log({"sample_inference": gen_text}, step=global_step)

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        if save_checkpoint is not None:
            save_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step
            }
            torch.save(save_dict, save_checkpoint)
            print(f"Interrupted training state saved to {save_checkpoint}")
        wandb.finish()
        return model, tokenizer

    # Save final checkpoint
    if save_checkpoint is not None:
        print(f"Saving final checkpoint to {save_checkpoint}")
        save_dict = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epochs,
            "global_step": global_step
        }
        torch.save(save_dict, save_checkpoint)
        # Optionally log checkpoint to W&B
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{wandb.run.id}",
            type="model",
            description="Final model checkpoint"
        )
        artifact.add_file(save_checkpoint)
        wandb.log_artifact(artifact)

    wandb.finish()
    return model, tokenizer


###############################################################################
# 6) VISUALIZATION (TORCHVIZ)
###############################################################################

def visualize_model(model, input_shape=(2, 64), vocab_size=50257, filename="transformer_architecture"):
    """
    Generate a visualization of the model using torchviz.
    """
    dummy_input = torch.randint(0, vocab_size, input_shape)
    logits = model(dummy_input)
    dot = make_dot(logits, params=dict(model.named_parameters()))
    dot.render(filename, format="pdf")
    print(f"Model visualization saved as '{filename}.pdf'")


###############################################################################
# 7) MAIN SCRIPT / CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference", "visualize"],
                        help="Mode of operation: train, inference, or visualize.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training/validation.")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (for truncation).")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of Transformer layers.")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Dimension of feedforward layers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--load_checkpoint", type=str, default="transformer_checkpoint.pt",
                        help="Path to checkpoint file to resume training or use for inference.")
    parser.add_argument("--save_checkpoint", type=str, default="transformer_checkpoint.pt",
                        help="Path to save final (or periodic) checkpoints.")
    parser.add_argument("--checkpoint_every", type=int, default=1000,
                        help="Save checkpoint every N steps (if set).")
    parser.add_argument("--sample_prompt", type=str, default="Hello how are",
                        help="Prompt text for generation during/after training.")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Max new tokens to generate at inference.")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging.")

    # NEW ARGUMENTS FOR MID-EPOCH VALIDATION
    parser.add_argument("--validate_after", type=int, default=1000,
                        help="Number of training steps after which to run a partial validation.")
    parser.add_argument("--val_subset", type=float, default=0.1,
                        help="Subset of validation data to use for mid-epoch validation. "
                             "If < 1.0, treated as fraction; if >= 1, treated as absolute number.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prepare the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 has no official pad_token; use eos_token for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Build the model (using GPT-2 pretrained embeddings)
    vocab_size = tokenizer.vocab_size  # ~50257 for GPT-2
    model = GenericTransformer(
        vocab_size=vocab_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        gpt2_model_name="gpt2"
    )

    ###########################################################################
    # MODE: TRAIN
    ###########################################################################
    if args.mode == "train":
        train_loader, full_val_loader, tokenizer = prepare_data(
            load_dataset_fn=load_wikitext103,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len
        )

        # Prepare a partial validation loader if needed
        partial_val_loader = None
        if args.validate_after is not None and args.validate_after > 0:
            val_dataset = full_val_loader.dataset
            full_size = len(val_dataset)
            if args.val_subset < 1.0:
                # treat as fraction
                subset_size = int(full_size * args.val_subset)
            else:
                # treat as absolute number
                subset_size = int(args.val_subset)
            subset_size = max(1, min(subset_size, full_size))

            indices = list(range(subset_size))
            partial_val_dataset = Subset(val_dataset, indices)
            partial_val_loader  = DataLoader(partial_val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)

        model, tokenizer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=full_val_loader,
            tokenizer=tokenizer,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            disable_wandb=args.disable_wandb,
            load_checkpoint=args.load_checkpoint,
            save_checkpoint=args.save_checkpoint,
            checkpoint_every=args.checkpoint_every,
            sample_prompt=args.sample_prompt,
            validate_after=args.validate_after,
            partial_val_loader=partial_val_loader
        )

    ###########################################################################
    # MODE: INFERENCE
    ###########################################################################
    elif args.mode == "inference":
        # Load model from checkpoint
        if args.load_checkpoint and os.path.isfile(args.load_checkpoint):
            print(f"Loading checkpoint for inference: {args.load_checkpoint}")
            ckpt = torch.load(args.load_checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
        else:
            raise FileNotFoundError("Must specify --load_checkpoint for inference.")

        print("Entering interactive inference mode. Type 'quit' to exit.\n")
        print("Note: Higher temperature (e.g. 0.8-1.0) = more random/creative")
        print("      Lower temperature (e.g. 0.1-0.4) = more focused/deterministic\n")

        while True:
            user_prompt = input("User prompt: ")
            if user_prompt.strip().lower() == "quit":
                print("Exiting inference mode.")
                break

            temp = input("Temperature (0.1-1.0, default=0.8): ").strip()
            temperature = float(temp) if temp else 0.8

            output_text = sample_inference(
                model,
                tokenizer,
                prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                temperature=temperature
            )

            print("Model response:", output_text)
            print("-" * 50)

    ###########################################################################
    # MODE: VISUALIZE
    ###########################################################################
    elif args.mode == "visualize":
        # Just visualize the architecture
        visualize_model(
            model, 
            input_shape=(2, args.seq_len),
            vocab_size=tokenizer.vocab_size,
            filename="transformer_architecture"
        )
