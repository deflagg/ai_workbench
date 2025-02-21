import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import warnings
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import GPT2Tokenizer
import logging

# Configure logging for the data pipeline
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Initialize the GPT-2 tokenizer and add special tokens if missing
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info("Added pad token to GPT-2 tokenizer.")
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    logger.info("Added BOS token to GPT-2 tokenizer.")

# ------------------------------------------------------------------
# DATA PIPELINE: Using WikiText-2 with a GPT-2 Tokenizer and Sliding Window
# ------------------------------------------------------------------

class WikiText2Dataset(Dataset):
    def __init__(self, split="train", max_seq_len=128, stride=64):
        """
        Loads WikiText-2 (raw) and creates samples using a sliding window approach.
        Each sample is a sequence of length max_seq_len tokens with a [BOS] token prepended.
        For causal language modeling, the input is tokens[:-1] and the target is tokens[1:].
        """
        logger.info(f"Loading WikiText-2 dataset for split: {split}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all non-empty lines into one long string
        all_text = " ".join([line["text"].strip() for line in dataset if line["text"].strip()])
        # Tokenize the concatenated text without adding special tokens
        tokens = tokenizer.encode(all_text, add_special_tokens=False)
        self.samples = []
        bos_token_id = tokenizer.bos_token_id
        # Create sliding windows with stride; reserve one token for BOS
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i: i + max_seq_len - 1]
            if len(window_tokens) < 1:
                continue
            # Prepend BOS token
            sample = [bos_token_id] + window_tokens
            if len(sample) < 2:
                continue
            self.samples.append(sample)
        logger.info(f"Finished processing split '{split}': {len(self.samples)} valid samples.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # For causal LM: input is tokens[:-1], target is tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long))

def collate_fn_wikitext(batch):
    """
    Pads sequences in a batch using the tokenizer's pad token.
    For targets, pads with -100 so that these positions are ignored in loss computation.
    """
    logger.debug(f"Collating a batch of {len(batch)} samples.")
    pad_id = tokenizer.pad_token_id
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded_inputs, padded_targets = [], []
    for inp, tgt in zip(inputs, targets):
        pad_length = max_len - len(inp)
        padded_inp = torch.cat([inp, torch.full((pad_length,), pad_id, dtype=torch.long)])
        padded_tgt = torch.cat([tgt, torch.full((pad_length,), -100, dtype=torch.long)])
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# ------------------------------------------------------------------
# SAMPLING: Top-K and Top-P Filtering Function
# ------------------------------------------------------------------

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):
    """
    Filter logits using top-k and nucleus (top-p) filtering.
    """
    # Top-K filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# ------------------------------------------------------------------
# MODEL: Decoder Block & Decoder-Only LM
# ------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.5):
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, causal_mask=None):
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + attn_output
        residual = x
        x = self.ln2(x)
        ff_output = self.ff(x)
        x = residual + ff_output
        return x

class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, 
                 dim_feedforward=1024, max_seq_length=128, dropout=0.5):
        super(DecoderOnlyLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        causal_mask = torch.triu(torch.full((seq_length, seq_length), float('-inf'), device=x.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

# ------------------------------------------------------------------
# INFERENCE FUNCTION WITH TOP-K and TOP-P SAMPLING
# ------------------------------------------------------------------

def generate_text(model, prompt, max_length=128, temperature=1.0, top_k=50, top_p=0.9, device=torch.device('cpu')):
    """
    Generates text using the model given a prompt, applying top-k and top-p sampling.
    """
    try:
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        model.eval()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_tokens:
            raise ValueError("Prompt must contain at least one valid token.")
        input_ids = prompt_tokens.copy()
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(max_length - len(input_ids)):
                logits = model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                # Apply top-k and top-p filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits.unsqueeze(0), top_k=top_k, top_p=top_p).squeeze(0)
                probs = torch.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                input_ids.append(next_token)
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = tokenizer.decode(input_ids)
        return generated

    except Exception as e:
        raise RuntimeError("An error occurred during text generation.") from e

# ------------------------------------------------------------------
# TRAINING FUNCTION WITH EARLY STOPPING AND 1-CYCLE LR POLICY
# ------------------------------------------------------------------

def train_model(model, train_dataloader, val_dataloader=None, num_epochs=10, max_lr=1e-3,
                device=torch.device('cpu'), checkpoint_interval=None, checkpoint_dir=None,
                inference_prompt="Once upon a time", max_seq_length=128, patience=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=max_lr/25)  # initial lr based on div_factor (default=25)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Baseline inference before training.
    print("\nBaseline inference before training:")
    baseline_output = generate_text(model, inference_prompt, max_length=max_seq_length, device=device)
    print("Baseline output:", baseline_output, "\n")
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    current_epoch = 0
    try:
        for epoch in range(num_epochs):
            current_epoch = epoch + 1
            model.train()
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
                # Log batch metrics to W&B every 10 batches
                if batch_idx % 10 == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx + len(train_dataloader) * epoch,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                    
            avg_loss = total_loss / len(train_dataloader)
            metrics = {"train_loss": avg_loss, "epoch": current_epoch, "lr": optimizer.param_groups[0]['lr']}
            if val_dataloader is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        val_loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                metrics["val_loss"] = avg_val_loss
                print(f"Timestamp: {datetime.datetime.now()} - Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping triggered at epoch {current_epoch}.")
                        wandb.log({"early_stopping_epoch": current_epoch})
                        return model
            else:
                print(f"Timestamp: {datetime.datetime.now()} - Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f}")
            
            wandb.log(metrics)
            
            if checkpoint_interval is not None and current_epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{current_epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                inference_output = generate_text(model, inference_prompt, max_length=max_seq_length, device=device)
                print("Inference output at checkpoint:", inference_output, "\n")
                wandb.log({
                    "generated_text": wandb.Html(f"<p><strong>Epoch {current_epoch}:</strong> {inference_output}</p>")
                })
        return model

    except KeyboardInterrupt:
        wandb.finish()
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_interrupt_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nKeyboardInterrupt detected. Checkpoint saved at {checkpoint_path}")
        sys.exit(0)

    except Exception as e:
        wandb.finish()
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_exception_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Exception occurred during training. Checkpoint saved at {checkpoint_path}")
        raise e

# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on the Decoder-Only LM using WikiText-2.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode: 'train' to train the model, 'inference' to generate text.")
    parser.add_argument("--model_path", type=str, default="wikitext2_decoder_only_model.pt",
                        help="Path to save/load the model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for text generation.")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Max sequence length for the model and dataset.")
    parser.add_argument("--stride", type=int, default=64,
                        help="Stride for the sliding window in dataset creation.")
    
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save a checkpoint every N epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, default="wikitext2-enhanced",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/username")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Dimension of the token and positional embeddings.")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads in the decoder blocks.")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of decoder blocks.")
    parser.add_argument("--dim_feedforward", type=int, default=128,
                        help="Dimension of the feedforward network in the decoder blocks.")
    parser.add_argument("--max_lr", type=float, default=3e-3,
                        help="Maximum learning rate for 1-cycle LR policy.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience based on validation loss.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == "train":
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        config = {
            "max_lr": args.max_lr,
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "max_seq_length": args.max_seq_length,
            "vocab_size": tokenizer.vocab_size
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"run_{wandb.util.generate_id()}"
        )

        # Use WikiText-2 for training and validation with sliding window
        train_dataset = WikiText2Dataset(split="train", max_seq_len=args.max_seq_length, stride=args.stride)
        val_dataset = WikiText2Dataset(split="validation", max_seq_len=args.max_seq_length, stride=args.stride)
        logger.info(f"Training dataset contains {len(train_dataset)} samples.")
        logger.info(f"Validation dataset contains {len(val_dataset)} samples.")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_wikitext)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_wikitext)
        
        # Initialize model with unified max_seq_length
        model = DecoderOnlyLM(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            max_seq_length=args.max_seq_length
        )
        
        # Resume from checkpoint if provided
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
            print(f"Resumed model from checkpoint {args.resume_checkpoint}")
        
        wandb.watch(model, log="all")
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        print("Starting training...")
        try:
            train_model(model, train_dataloader, val_dataloader,
                        num_epochs=args.num_epochs,
                        max_lr=args.max_lr,
                        device=device,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_dir=args.checkpoint_dir,
                        inference_prompt=args.prompt,
                        max_seq_length=args.max_seq_length,
                        patience=args.patience)
        except Exception as e:
            print(f"Training terminated with an exception: {str(e)}")
            wandb.finish()
            return
        
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        wandb.finish()

    elif args.mode == "inference":
        try:
            model = DecoderOnlyLM(
                vocab_size=len(tokenizer),
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dim_feedforward=args.dim_feedforward,
                max_seq_length=args.max_seq_length
            )
            if os.path.exists(args.model_path):
                model.load_state_dict(torch.load(args.model_path, map_location=device))
                model.to(device)
                print(f"Model loaded from {args.model_path}")
            else:
                print(f"Model path {args.model_path} does not exist. Please train the model first.")
                return
            stop_word = "exit"
            print(f"Type '{stop_word}' at the prompt to quit inference mode.")
            while True:
                try:
                    user_prompt = input("\nEnter a prompt for text generation (or press Enter to use the default prompt): ")
                    if user_prompt.strip().lower() == stop_word:
                        print("Exiting inference mode.")
                        break
                    if user_prompt.strip() == "":
                        user_prompt = args.prompt
                    generated_text = generate_text(model, user_prompt, max_length=args.max_seq_length, device=device)
                    print("Generated Text:", generated_text)
                except Exception as e:
                    print(f"Error during text generation: {str(e)}")
                    print("Please try again with a different prompt.")
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            print("Please try again with a different prompt.")

if __name__ == "__main__":
    main()
