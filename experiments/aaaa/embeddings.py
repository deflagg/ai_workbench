import os
import argparse
import warnings
import logging
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer

import wandb

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Load environment variables and ignore some warnings
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
# DATA PIPELINE: WikiText-103 Dataset with Sliding Window
# ------------------------------------------------------------------

class WikiText2Dataset(Dataset):
    def __init__(self, split="train", max_seq_len=128, stride=64):
        """
        Loads WikiText-103 (raw) and creates samples using a sliding window approach.
        Each sample is a sequence of length max_seq_len tokens with a [BOS] token prepended.
        For causal language modeling, the input is tokens[:-1] and the target is tokens[1:].
        """
        logger.info(f"Loading WikiText-103 dataset for split: {split}")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
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
    Returns padded inputs, padded targets, and a padding mask.
    """
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
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)
    padding_mask = (padded_inputs == pad_id)
    return padded_inputs, padded_targets, padding_mask

# ------------------------------------------------------------------
# MODEL: Embedding-Only Language Model
# ------------------------------------------------------------------

class EmbeddingOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_seq_length=128, dropout=0.1):
        """
        A minimal language model that only learns token and positional embeddings,
        followed by a linear projection to predict the next token.
        """
        super(EmbeddingOnlyLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

# ------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------

def train_model(model, train_dataloader, num_epochs=10, learning_rate=1e-3, device=torch.device('cpu'), val_dataloader=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        # Training loop over batches
        for batch_idx, (inputs, targets, _) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Log batch loss to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx
            })
        
        avg_train_loss = total_loss / len(train_dataloader)
        # Log training loss for the epoch
        wandb.log({
            "training_loss": avg_train_loss,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
        
        # If a validation dataloader is provided, evaluate the model
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets, _ in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            wandb.log({
                "validation_loss": avg_val_loss,
                "epoch": epoch + 1
            })
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
            model.train()
            
    return model

# ------------------------------------------------------------------
# INFERENCE FUNCTION: Greedy text generation
# ------------------------------------------------------------------

def generate_text(model, tokenizer, prompt, max_length, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Generate tokens until reaching max_length
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(input_ids)
            # Take the logits of the last token and pick the highest probability token
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # If the model predicts the end-of-sequence token, stop early (if defined)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    generated_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    return generated_text

# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train embeddings using a simplified language model or generate text.")
    parser.add_argument("--model_path", type=str, default="embeddings_model.pt", help="Path to save/load the model.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length for the model and dataset.")
    parser.add_argument("--stride", type=int, default=64, help="Stride for the sliding window in dataset creation.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the embeddings.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode: 'train' to train the model, 'inference' to generate text.")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt text for inference mode.")
    args = parser.parse_args()

    # Initialize wandb logging
    wandb.init(project="embedding-only-lm", config=vars(args))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == "train":
        # Prepare training dataset and dataloader
        train_dataset = WikiText2Dataset(split="train", max_seq_len=args.max_seq_length, stride=args.stride)
        print(f"Training dataset contains {len(train_dataset)} samples.")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_wikitext)
        
        # Prepare validation dataset and dataloader
        val_dataset = WikiText2Dataset(split="validation", max_seq_len=args.max_seq_length, stride=args.stride)
        print(f"Validation dataset contains {len(val_dataset)} samples.")
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_wikitext)

        # Initialize the embedding-only model
        model = EmbeddingOnlyLM(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            dropout=args.dropout
        )
        model.to(device)

        print("Starting training...")
        # Pass the validation dataloader to train_model
        model = train_model(model, train_dataloader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=device, val_dataloader=val_dataloader)

        # Save the trained model
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        wandb.save(args.model_path)
    
    elif args.mode == "inference":
        # Initialize and load the model for inference
        model = EmbeddingOnlyLM(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            dropout=args.dropout
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Loaded model for inference.")

        # Generate text from the prompt
        generated_text = generate_text(model, tokenizer, args.prompt, max_length=args.max_seq_length, device=device)
        print("Generated Text:\n", generated_text)
        wandb.log({"generated_text": generated_text})

    wandb.finish()

if __name__ == "__main__":
    main()
