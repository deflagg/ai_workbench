import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output.log", sys.stdout)

# 1. Define the Vocabulary and Mappings with a dedicated padding symbol '#'
vocab = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
         ]
letters = vocab[1:27]
token2idx = {token: i for i, token in enumerate(vocab)}
idx2token = {i: token for token, i in token2idx.items()}

# 2. Data Generation (according to the grammar)
# A global counter to vary the output deterministically.
counter = 0

def generate_valid_string(max_seq_len=7):
    """
    Generate a string of fixed length 'seq_len' that follows a deterministic pattern.
    
    The string is built by interleaving two arithmetic sequences:
      - Characters at even positions come from an arithmetic progression with
        starting index 'start_even' and step 'offset_even'.
      - Characters at odd positions come from another arithmetic progression with
        starting index 'start_odd' and step 'offset_odd'.
        
    Both progressions wrap around (using modulo arithmetic) in the allowed letters.
    The parameters are chosen based on a global counter so that every call produces
    a new (yet fully predictable) string.
    """
    global counter
    # Determine parameters from the counter.
    start_even = counter % len(letters)
    offset_even = ((counter // len(letters)) % len(letters)) + 1  # step at least 1
    start_odd = (counter + 3) % len(letters)
    offset_odd = (((counter + 3) // len(letters)) % len(letters)) + 1
    counter += 1  # Increment counter for the next call.
    
    result = []
    # Build the string by interleaving positions.
    for i in range(max_seq_len):
        if i % 2 == 0:
            # Even index: use the even arithmetic progression.
            letter_index = (start_even + (i // 2) * offset_even) % len(letters)
        else:
            # Odd index: use the odd arithmetic progression.
            letter_index = (start_odd + (i // 2) * offset_odd) % len(letters)
        result.append(letters[letter_index])
    return ''.join(result)


class LanguageDataset(Dataset):
    def __init__(self, num_samples=10000, max_seq_len=3):
        """
        num_samples: how many samples to generate
        max_seq_len: maximum length for each generated sequence
        """
        self.samples = [generate_valid_string(max_seq_len) for _ in range(num_samples)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Tokenize the string into indices.
        token_ids = [token2idx[t] for t in sample]
        # For language modeling, the input is the sequence except the last token,
        # and the target is the sequence shifted by one.
        return (torch.tensor(token_ids[:-1], dtype=torch.long),
                torch.tensor(token_ids[1:], dtype=torch.long))

def collate_fn(batch):
    # Collate a batch of (input, target) pairs, padding to the same length.
    inputs, targets = zip(*batch)
    max_seq_len = max(len(x) for x in inputs)
    padded_inputs = []
    padded_targets = []
    pad_token_idx = token2idx['#']
    for inp, tgt in zip(inputs, targets):
        pad_length = max_seq_len - len(inp)
        padded_inputs.append(torch.cat([
            inp,
            torch.full((pad_length,), pad_token_idx, dtype=torch.long)
        ]))
        padded_targets.append(torch.cat([
            tgt,
            torch.full((pad_length,), -100, dtype=torch.long)  # -100 will be ignored in loss.
        ]))
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# 3. Define the Transformer Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, 
                 dim_feedforward=64, max_seq_length=20, dropout=0.5):
        super(TransformerLM, self).__init__()
        # Set the padding_idx to the index of '#' in our vocabulary.
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=token2idx['#'])
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        # Create the Transformer encoder with batch_first=True.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
    
    def forward(self, x):
        """
        x: (batch_size, seq_length)
        Returns logits: (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.pos_embedding(positions)
        x_emb = self.token_embedding(x) + pos_emb

        # Create key padding mask (True for positions that are pad tokens)
        key_padding_mask = (x == self.token_embedding.padding_idx)

        # Create causal mask (True where attention should be masked).
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device).bool()

        x_transformed = self.transformer(x_emb, mask=mask, src_key_padding_mask=key_padding_mask)
        logits = self.fc_out(x_transformed)
        return logits

# 4. Training the Model with Validation and Checkpointing
def train_model(model, train_dataloader, val_dataloader=None, num_epochs=10, lr=1e-3,
                device=torch.device('cpu'), checkpoint_interval=None, checkpoint_dir=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    current_epoch = 0
    try:
        for epoch in range(num_epochs):
            current_epoch = epoch + 1
            model.train()
            total_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            
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
                print(f"Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f}")
            
            # Save checkpoint at every specified interval.
            if checkpoint_interval is not None and current_epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{current_epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
        return model

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully: save checkpoint and exit.
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_interrupt_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nKeyboardInterrupt detected. Checkpoint saved at {checkpoint_path}")
        # Exit the program gracefully.
        sys.exit(0)

    except Exception as e:
        # Save a checkpoint if any other exception occurs.
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_exception_epoch_{current_epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Exception occurred during training. Checkpoint saved at {checkpoint_path}")
        raise e


# 5. Inference (Text Generation) with Enhanced Exception Handling
def generate_text(model, prompt, max_length=20, temperature=1.0, device=torch.device('cpu')):
    """
    Generates text from a given prompt, up to max_length.
    If temperature <= 0, raises ValueError.
    """
    try:
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")

        model.eval()
        # Convert prompt characters to indices (ignoring any character not in the vocabulary)
        input_ids = [token2idx[t] for t in prompt if t in token2idx]
        if not input_ids:
            raise ValueError("Prompt must contain at least one valid token from the vocabulary.")

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Generate tokens until reaching max_length.
            for _ in range(max_length - len(input_ids)):
                logits = model(input_tensor)
                # Get logits for the last token and adjust by temperature.
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                input_ids.append(next_token)
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Remove pad tokens from the final string.
        generated = ''.join([idx2token[i] for i in input_ids if i != token2idx['#']])
        return generated

    except Exception as e:
        # Re-raise the error with additional context.
        raise RuntimeError("An error occurred during text generation.") from e

# 6. Main function for command line interface
def main():
    parser = argparse.ArgumentParser(description="Train or run inference on the Transformer LM.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode: 'train' to train the model, 'inference' to generate text.")
    parser.add_argument("--model_path", type=str, default="transformer_model.pt",
                        help="Path to save/load the model.")
    parser.add_argument("--prompt", type=str, default="A",
                        help="Prompt for text generation (used in inference mode).")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs (used in train mode).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (used in train mode).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (used in train mode).")
    parser.add_argument("--max_seq_length", type=int, default=300,
                        help="Max sequence length for the model (used in train and inference modes).")
    parser.add_argument("--train_samples", type=int, default=3000,
                        help="Number of training samples (used in train mode).")
    parser.add_argument("--val_samples", type=int, default=100,
                        help="Number of validation samples (used in train mode).")
    # New arguments for model dimensions.
    parser.add_argument("--d_model", type=int, default=32,
                        help="Dimension of the token and positional embeddings.")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads in the Transformer encoder.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of Transformer encoder layers.")
    parser.add_argument("--dim_feedforward", type=int, default=64,
                        help="Dimension of the feedforward network in the Transformer encoder layers.")
    # New arguments for checkpointing.
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save a checkpoint every N epochs during training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    args = parser.parse_args()

    # Use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == "train":
        # Create training and validation datasets and DataLoaders using max_seq_length
        train_dataset = LanguageDataset(num_samples=args.train_samples, max_seq_len=args.max_seq_length)
        val_dataset = LanguageDataset(num_samples=args.val_samples, max_seq_len=args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Instantiate the model with the provided hyperparameters.
        model = TransformerLM(
            vocab_size=len(vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            max_seq_length=args.max_seq_length
        )
        
        # Create the checkpoint directory if it does not exist.
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        
        print("Starting training...")
        try:
            train_model(model, train_dataloader, val_dataloader,
                        num_epochs=args.num_epochs,
                        lr=args.lr,
                        device=device,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_dir=args.checkpoint_dir)
        except Exception as e:
            print(f"Training terminated with an exception: {str(e)}")
            # Optionally, you can exit or perform additional cleanup here.
            return
        
        # Save the final trained model.
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "inference":
        try:
            # Instantiate the model and load saved parameters.
            model = TransformerLM(
                vocab_size=len(vocab),
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
    
            # Inference mode: allow interactive prompt generation.
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
    # Wait for user input to exit.
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
