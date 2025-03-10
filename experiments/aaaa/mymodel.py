import datetime
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import warnings
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import GPT2Tokenizer
import logging

# ------------------------------------------------------------------------------
# LOGGING & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# TOKENIZER INITIALIZATION
# ------------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info("Added pad token to GPT-2 tokenizer.")
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    logger.info("Added BOS token to GPT-2 tokenizer.")

# ------------------------------------------------------------------------------
# DATA PIPELINE
# ------------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split="train", max_seq_len=128, stride=64, dataset_name="wikitext-103-raw-v1"):
        logger.info(f"Loading {dataset_name} dataset for split: {split}")
        dataset = load_dataset("wikitext", dataset_name, split=split)
        all_text = " ".join([line["text"].strip() for line in dataset if line["text"].strip()])
        tokens = tokenizer.encode(all_text, add_special_tokens=False)
        self.samples = []
        bos_token_id = tokenizer.bos_token_id
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i: i + max_seq_len - 1]  # reserve one slot for BOS
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
        # Use all tokens except the last as input; the target is the final token.
        input_ids = tokens[:-1]
        target_id = tokens[-1]
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.long))

def collate_fn_token_navigator(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id
    padded_inputs = []
    for inp in inputs:
        pad_length = max_len - len(inp)
        padded_inp = torch.cat([inp, torch.full((pad_length,), pad_id, dtype=torch.long)])
        padded_inputs.append(padded_inp)
    padded_inputs = torch.stack(padded_inputs)
    targets = torch.stack(targets)
    return padded_inputs, targets, lengths

# ------------------------------------------------------------------------------
# MODEL: TOKENSPACE NAVIGATOR
# ------------------------------------------------------------------------------
class TokenSpaceNavigator(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_seq_length=128):
        """
        The model consists of token and positional embeddings and a single linear layer (q_layer)
        that computes a query vector from the last token's embedding. The query is then compared
        (via dot-product) against the token embeddings to yield logits for the next token.
        """
        super(TokenSpaceNavigator, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        self.q_layer = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, lengths):
        """
        x: (batch_size, seq_length) of token IDs (padded)
        lengths: (batch_size,) with the actual lengths of each input sequence
        """
        batch_size, seq_length = x.shape
        token_emb = self.token_embedding(x)  # (batch, seq_length, d_model)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.pos_embedding(positions)
        emb = token_emb + pos_emb
        # Gather the last valid token from each sample using lengths.
        last_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, emb.size(2))
        last_emb = emb.gather(1, last_indices).squeeze(1)  # (batch, d_model)
        q = self.q_layer(last_emb)  # (batch, d_model)
        logits = torch.matmul(q, self.token_embedding.weight.transpose(0, 1))  # (batch, vocab_size)
        return logits

# ------------------------------------------------------------------------------
# TEXT GENERATION FUNCTION (WITHOUT TOP-K / TOP-P FILTERING)
# ------------------------------------------------------------------------------
def generate_text(model, prompt, max_length=128, temperature=1.0, device=torch.device('cpu')):
    """
    Generates text using the model. The prompt is tokenized and then the model iteratively
    predicts the next token using the TokenSpaceNavigator.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be a positive number.")
    model.eval()
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least one valid token.")
    input_ids = prompt_tokens.copy()
    with torch.no_grad():
        for _ in range(max_length - len(input_ids)):
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            lengths = torch.tensor([len(input_ids)], device=device)
            logits = model(input_tensor, lengths)  # (1, vocab_size)
            next_token_logits = logits[0] / temperature
            probs = torch.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            input_ids.append(next_token)
    generated = tokenizer.decode(input_ids)
    return generated

# ------------------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------------------
def train_model(model, train_dataloader, val_dataloader=None, num_epochs=10,
                device=torch.device('cpu'), checkpoint_interval=None, checkpoint_dir=None,
                inference_prompt="Once upon a time", max_seq_length=128, patience=10,
                save_checkpoints_to_wandb=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
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
            for batch_idx, (inputs, targets, lengths) in enumerate(train_dataloader):
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                optimizer.zero_grad()
                logits = model(inputs, lengths)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 10 == 0:
                    wandb.log({"batch_loss": loss.item(), "batch": batch_idx + len(train_dataloader)*epoch})
            avg_loss = total_loss / len(train_dataloader)
            metrics = {"train_loss": avg_loss, "epoch": current_epoch}
            if val_dataloader is not None:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, lengths in val_dataloader:
                        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                        logits = model(inputs, lengths)
                        val_loss = criterion(logits, targets)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                metrics["val_loss"] = avg_val_loss
                print(f"{datetime.datetime.now()} - Epoch {current_epoch}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
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
                print(f"{datetime.datetime.now()} - Epoch {current_epoch}/{num_epochs} - Train Loss: {avg_loss:.4f}")
            
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
                if save_checkpoints_to_wandb:
                    wandb.save(checkpoint_path)
        return model

    except KeyboardInterrupt:
        wandb.finish()
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_interrupt_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nKeyboardInterrupt detected. Checkpoint saved at {checkpoint_path}")
        if save_checkpoints_to_wandb:
            wandb.save(checkpoint_path)
        sys.exit(0)

    except Exception as e:
        wandb.finish()
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_exception_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Exception occurred during training. Checkpoint saved at {checkpoint_path}")
        if save_checkpoints_to_wandb:
            wandb.save(checkpoint_path)
        raise e

# ------------------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or run inference on the TokenSpaceNavigator using WikiText.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode: 'train' to train the model, 'inference' to generate text.")
    parser.add_argument("--model_path", type=str, default="token_space_navigator_model.pt",
                        help="Path to save/load the model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for text generation.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Max sequence length for the model and dataset.")
    parser.add_argument("--stride", type=int, default=64,
                        help="Stride for the sliding window in dataset creation.")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Save a checkpoint every N epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, default="TokenSpaceNavigator Training",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/username")
    parser.add_argument("--wandb_save_checkpoints", type=bool, default=False,
                        help="If set, checkpoints will also be saved to W&B.")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Dimension of the token and positional embeddings.")
    parser.add_argument("--wikitext_name", type=str, default="wikitext-2-raw-v1",
                        help="Name of the WikiText dataset to use (e.g., 'wikitext-2-raw-v1' or 'wikitext-103-raw-v1').")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == "train":
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        config = {
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "d_model": args.d_model,
            "max_seq_length": args.max_seq_length,
            "vocab_size": tokenizer.vocab_size,
            "wikitext_name": args.wikitext_name
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"run_{wandb.util.generate_id()}"
        )

        # Prepare training and validation datasets using the selected WikiText dataset name.
        train_dataset = WikiTextDataset(split="train", max_seq_len=args.max_seq_length, stride=args.stride, dataset_name=args.wikitext_name)
        val_dataset = WikiTextDataset(split="validation", max_seq_len=args.max_seq_length, stride=args.stride, dataset_name=args.wikitext_name)
        logger.info(f"Training dataset contains {len(train_dataset)} samples.")
        logger.info(f"Validation dataset contains {len(val_dataset)} samples.")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_token_navigator)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_token_navigator)
        
        model = TokenSpaceNavigator(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            max_seq_length=args.max_seq_length
        )
        
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device))
            print(f"Resumed model from checkpoint {args.resume_checkpoint}")
        
        model.to(device)
        
        wandb.watch(model, log="all")
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        print("Starting training...")
        try:
            train_model(model, train_dataloader, val_dataloader,
                        num_epochs=args.num_epochs,
                        device=device,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_dir=args.checkpoint_dir,
                        inference_prompt=args.prompt,
                        max_seq_length=args.max_seq_length,
                        patience=10,
                        save_checkpoints_to_wandb=args.wandb_save_checkpoints)
        except Exception as e:
            print(f"Training terminated with an exception: {str(e)}")
            wandb.finish()
            return
        
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        wandb.finish()

    elif args.mode == "inference":
        try:
            model = TokenSpaceNavigator(
                vocab_size=len(tokenizer),
                d_model=args.d_model,
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

if __name__ == "__main__":
    main()
