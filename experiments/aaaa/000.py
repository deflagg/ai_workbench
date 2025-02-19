import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import random
import warnings
import re
import wandb
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# ============== DATA GENERATION (IMPROVED) ==============
# The following version removes nonsense tokens like 'xek', 'zi', 'ro', 'nox', etc.
# and includes only common English words. This should produce more coherent sentences,
# though they are still randomly generated.

# -----------------------------------------------------------------------------------
# 1) Complex Vocabulary
# -----------------------------------------------------------------------------------
nouns = [
    "cat", "kitty", "dog", "hound", "puppy", "rat", "mouse",
    "house", "car", "ship", "building", "book", "pen", "computer", "phone",
    "tree", "forest", "river", "hill", "mountain", "flower", "garden", "bee",
    "cake", "spoon", "fork", "plate", "guitar", "song", "cloud", "town"
]

verbs = [
    "eat", "run", "jump", "see", "find", "hold", "dance", "sing", "explore",
    "destroy", "build", "follow", "watch", "chase", "catch", "paint"
]

adjectives = [
    "big", "small", "tiny", "massive", "green", "blue", "amazing",
    "mysterious", "tasty", "fragile", "happy", "sad", "quick"
]

adverbs = [
    "quickly", "slowly", "softly", "loudly", "brightly",
    "quietly", "suddenly", "calmly"
]

determiners = [
    "the", "a", "an", "that", "this", "your", "my", "their"
]

# Natural conjunctions
conjs = [
    "and", "but", "or", "so", "because"
]

# Punctuation
punct = [".", "!", "?", "..."]

# -----------------------------------------------------------------------------------
# 2) Word-Level Vocabulary
# -----------------------------------------------------------------------------------
word_list = nouns + verbs + adjectives + adverbs + determiners + conjs + punct
word_list = list(set(word_list))   # remove duplicates if any
word_list.sort()                   # sort for a consistent ordering

PAD_TOKEN = "#"                    # We'll use "#" as our padding token
vocab = [PAD_TOKEN] + word_list    # final vocab includes PAD at index 0
token2idx = {token: i for i, token in enumerate(vocab)}
idx2token = {i: token for token, i in token2idx.items()}
PAD_IDX = token2idx[PAD_TOKEN]

# -----------------------------------------------------------------------------------
# 3) Grammar Functions
# -----------------------------------------------------------------------------------
def random_noun_phrase():
    """Noun phrase: optional determiner, 1-2 adjectives, then a noun."""
    phrase = []
    # 50% chance of a determiner
    if random.random() < 0.5:
        phrase.append(random.choice(determiners))
    # 40% chance to add 1-2 adjectives
    if random.random() < 0.4:
        for _ in range(random.randint(1, 2)):
            phrase.append(random.choice(adjectives))
    # always add a noun
    phrase.append(random.choice(nouns))
    return " ".join(phrase)

def random_verb_phrase():
    """
    Verb phrase:
      - 30% chance of adverb before verb
      - verb
      - 30% chance of adverb after verb
    """
    parts = []
    if random.random() < 0.3:
        parts.append(random.choice(adverbs))
    parts.append(random.choice(verbs))
    if random.random() < 0.3:
        parts.append(random.choice(adverbs))
    return " ".join(parts)

def random_subordinate_clause():
    """
    Creates a subordinate clause, e.g.:
      "if the cat runs quickly then the hound chases a mouse"
      "when my dog eats cake your rat jump"
    """
    starters = ["if", "when", "although", "because"]
    starter = random.choice(starters)
    s1 = f"{starter} {random_noun_phrase()} {random_verb_phrase()}"
    # 50% chance to add an object in the first part
    if random.random() < 0.5:
        s1 += " " + random_noun_phrase()
    # 50% chance we add "then" if the starter is "if"
    connector = ""
    if starter == "if" and random.random() < 0.6:
        connector = "then"
    # second clause
    s2 = f"{random_noun_phrase()} {random_verb_phrase()}"
    # 50% chance to add an object in second clause
    if random.random() < 0.5:
        s2 += " " + random_noun_phrase()
    # 50% chance to insert a random conjunction
    if random.random() < 0.5:
        conj = random.choice(conjs)
        s2 = conj + " " + s2
    return f"{s1} {connector} {s2}".strip()

def random_sentence():
    """A single independent clause with optional object."""
    subj = random_noun_phrase()
    vp = random_verb_phrase()
    sentence = [subj, vp]
    # 50% chance to add an object
    if random.random() < 0.5:
        sentence.append(random_noun_phrase())
    return " ".join(sentence)

def random_compound_sentence():
    """
    Returns one or two clauses, possibly subordinate, joined by a conj.
    Then adds random punctuation.
    """
    # First clause (40% chance subordinate)
    if random.random() < 0.4:
        s1 = random_subordinate_clause()
    else:
        s1 = random_sentence()
    # 50% chance to add a second clause
    if random.random() < 0.5:
        # 40% chance second is subordinate
        if random.random() < 0.4:
            s2 = random_subordinate_clause()
        else:
            s2 = random_sentence()
        # usually add a random conjunction
        conj = random.choice(conjs)
        s = f"{s1} {conj} {s2}"
    else:
        s = s1
    # Add final punctuation
    return s + random.choice(punct)

def tokenize(text):
    """
    Splits punctuation (including multi-char '...')
    so each punctuation mark becomes its own token.
    """
    text = re.sub(r"([.?!]+)", r" \1", text)
    tokens = text.split()
    return tokens


# ============== END DATA GENERATION ==============



# ============== DATASET AND COLLATION ==============
class LanguageDataset(Dataset):
    def __init__(self, num_samples=10000, max_seq_len=10):
        self.samples = []
        for _ in range(num_samples):
            sentence = random_compound_sentence()
            tokens = tokenize(sentence)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        token_ids = [token2idx[t] for t in tokens if t in token2idx]
        if len(token_ids) < 2:
            token_ids = token_ids + [PAD_IDX]*(2 - len(token_ids))
        return (torch.tensor(token_ids[:-1], dtype=torch.long),
                torch.tensor(token_ids[1:], dtype=torch.long))

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_seq_len = max(len(x) for x in inputs)
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_length = max_seq_len - len(inp)
        padded_inp = torch.cat([
            inp,
            torch.full((pad_length,), PAD_IDX, dtype=torch.long)
        ])
        padded_tgt = torch.cat([
            tgt,
            torch.full((pad_length,), -100, dtype=torch.long)
        ])
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# ============== DECODER BLOCK & DECODER-ONLY MODEL ==============
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
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
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, 
                dim_feedforward=64, max_seq_length=20, dropout=0.5):
        super(DecoderOnlyLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
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

def generate_text(model, prompt, max_length=20, temperature=1.0, device=torch.device('cpu')):
    try:
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        model.eval()
        prompt_tokens = tokenize(prompt)
        input_ids = [token2idx[t] for t in prompt_tokens if t in token2idx]
        if not input_ids:
            raise ValueError("Prompt must contain at least one valid token from the vocabulary.")
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(max_length - len(input_ids)):
                logits = model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                input_ids.append(next_token)
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = " ".join([idx2token[i] for i in input_ids if i != PAD_IDX])
        return generated

    except Exception as e:
        raise RuntimeError("An error occurred during text generation.") from e

def train_model(model, train_dataloader, val_dataloader=None, num_epochs=10, lr=1e-3,
                device=torch.device('cpu'), checkpoint_interval=None, checkpoint_dir=None,
                inference_prompt="ga cat", max_seq_length=20):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Baseline inference before any training has occurred.
    print("\nBaseline inference before training:")
    baseline_output = generate_text(model, inference_prompt, max_length=max_seq_length, device=device)
    print("Baseline output:", baseline_output, "\n")
    
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
                total_loss += loss.item()
                
                # Log batch metrics to W&B
                if batch_idx % 10 == 0:  # Log every 10 batches
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx + len(train_dataloader) * epoch
                    })
                    
            avg_loss = total_loss / len(train_dataloader)
            
            # Validation and logging
            metrics = {"train_loss": avg_loss, "epoch": current_epoch}
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
                print(f"Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {current_epoch}/{num_epochs} - Training Loss: {avg_loss:.4f}")
            
            # Log metrics to W&B
            wandb.log(metrics)
            
            if checkpoint_interval is not None and current_epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{current_epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                
                # Execute an inference at the checkpoint and log to W&B
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

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on the Decoder-Only LM.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode: 'train' to train the model, 'inference' to generate text.")
    parser.add_argument("--model_path", type=str, default="decoder_only_model.pt",
                        help="Path to save/load the model.")
    parser.add_argument("--prompt", type=str, default="ga cat",
                        help="Prompt for text generation (used in inference mode and for baseline/in-checkpoint inference).")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs (used in train mode).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (used in train mode).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (used in train mode).")
    parser.add_argument("--max_seq_length", type=int, default=50,
                        help="Max sequence length for the model (used in train and inference modes).")
    parser.add_argument("--train_samples", type=int, default=5000,
                        help="Number of training samples (used in train mode).")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Number of validation samples (used in train mode).")
    parser.add_argument("--d_model", type=int, default=8,
                        help="Dimension of the token and positional embeddings.")
    parser.add_argument("--nhead", type=int, default=1,
                        help="Number of attention heads in the decoder blocks.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of decoder blocks.")
    parser.add_argument("--dim_feedforward", type=int, default=32,
                        help="Dimension of the feedforward network in the decoder blocks.")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save a checkpoint every N epochs during training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--wandb_project", type=str, default="complex-grammer1-decoder-only",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/username")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.mode == "train":
        # Initialize W&B
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "max_seq_length": args.max_seq_length,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "device": str(device),
            "vocab_size": len(vocab)
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=f"run_{wandb.util.generate_id()}"
        )

        train_dataset = LanguageDataset(num_samples=args.train_samples, max_seq_len=args.max_seq_length)
        val_dataset = LanguageDataset(num_samples=args.val_samples, max_seq_len=args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = DecoderOnlyLM(
            vocab_size=len(vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            max_seq_length=args.max_seq_length
        )
        
        # Log model architecture to W&B
        wandb.watch(model, log="all")
        
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        
        print("Starting training...")
        try:
            train_model(model, train_dataloader, val_dataloader,
                        num_epochs=args.num_epochs,
                        lr=args.lr,
                        device=device,
                        checkpoint_interval=args.checkpoint_interval,
                        checkpoint_dir=args.checkpoint_dir,
                        inference_prompt=args.prompt,
                        max_seq_length=args.max_seq_length)
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
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
