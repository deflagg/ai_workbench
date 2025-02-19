from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model (e.g., BERT tokenizer and model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    """Step 1: Tokenize the input text"""
    inputs = tokenizer(text, return_tensors="pt")
    subwords = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
    print("Tokenized into:", subwords)
    return inputs, subwords

def get_embeddings(inputs):
    """Step 2: Get embeddings from the model"""
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print("Embeddings shape:", embeddings.shape)
    return embeddings

def print_results(subwords, embeddings, filename='embeddings.txt'):
    """Step 3: Save the results to a file"""
    with open(filename, 'w') as f:
        for subword, embedding in zip(subwords, embeddings[0]):
            f.write(f"{subword}\t{embedding.tolist()}\n")
    print(f"Embeddings saved to {filename}")

def main():
    text = "Transformers are powerful models for NLP."
    
    # Step 1: Tokenize
    inputs, subwords = tokenize_text(text)
    
    # Step 2: Get embeddings
    embeddings = get_embeddings(inputs)
    
    # Step 3: Save results
    print_results(subwords, embeddings)

if __name__ == "__main__":
    main()
