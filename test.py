import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import json

# Path to the .bin file of the pre-trained GPT-2 model
model_path = 'generator/gpt2/models/model_v5/story_gen.bin'
config_path = 'generator/gpt2/models/model_v5/config1.json'

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the configuration file
with open(config_path, "r") as config_file:
    config_dict = json.load(config_file)

# Initialize the configuration object
config = GPT2Config.from_dict(config_dict)

# Initialize the model with the configuration
model = GPT2LMHeadModel(config)

# Load the model weights from the .bin file
weights = torch.load(model_path, map_location='cpu')
model.load_state_dict(weights)

# Set the model to evaluation mode
model.eval()

# Function to generate text using the loaded model
def generate_text(seed_text, max_length=100):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example seed text
seed_text = "Hello, I'm a language model,"

# Generate text
generated_text = generate_text(seed_text)

# Print the generated text
print("Generated Text:")
print(generated_text)
