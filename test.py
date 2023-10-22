import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer (replace YourTokenizer() with your actual tokenizer)
tokenizer = YourTokenizer()  # Replace YourTokenizer() with the actual tokenizer you used during training

# Load the pre-trained model
model_path = 'generator/gpt2/models/model_v5'
model = tf.keras.models.load_model(model_path)


# Function to generate a story
def generate_story(seed_text, max_length=100):
    # Tokenize the seed text
    seed_text_encoded = tokenizer.texts_to_sequences([seed_text])[0]

    # Pad the encoded seed text
    seed_text_padded = pad_sequences([seed_text_encoded], maxlen=max_length, padding='pre')

    # Generate the story
    generated_text = []
    for _ in range(max_length):
        # Predict the next word
        predicted_word_index = model.predict_classes(seed_text_padded, verbose=0)[0]

        # Map the index back to the word using the tokenizer
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        # Break if the predicted word is not found or is an end token
        if predicted_word in ['<end>', '']:
            break

        # Add the predicted word to the generated text
        generated_text.append(predicted_word)

        # Update the seed text for the next iteration
        seed_text_encoded.append(predicted_word_index)
        seed_text_padded = pad_sequences([seed_text_encoded], maxlen=max_length, padding='pre')

    # Join the generated words to form the story
    generated_story = ' '.join(generated_text)
    return generated_story


# Example seed text
seed_text = "Once upon a time, there was a dragon"

# Generate the story
generated_story = generate_story(seed_text)

# Print the generated story
print("Generated Story:")
print(generated_story)
