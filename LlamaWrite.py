import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Set page configuration
st.set_page_config(page_title="Blog Content Generator")

# Ensure offload folder exists for large models
offload_folder = "./offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

# Authenticate with Hugging Face using your provided API token
login("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")

# Load the model (GPT-2 or another model)
@st.cache_resource(ttl=3600)  # Cache models for 1 hour
def load_models():
    try:
        model_name = "gpt2"  # Fallback model (Replace with any preferred model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        st.write(f"Model {model_name} loaded successfully.")
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Function to generate blog content based on the topic, word count, and target audience
def generate_blog_content(topic, word_count, target_audience):
    model, tokenizer = load_models()

    if model is None:
        return "Sorry, I couldn't generate personalized content at this time."

    try:
        # Set the prompt to give proper context to the model
        prompt = (
            f"Write a blog post about '{topic}', with a focus on providing relevant, "
            f"informative, and engaging content for a {target_audience} audience."
        )

        # Tokenize the prompt and generate the content
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate the text with the model
        output = model.generate(
            inputs["input_ids"],
            max_length=min(word_count * 5, 1024),  # Ensure we don't exceed token limit
            num_return_sequences=1,  # Return only one generated sequence
            do_sample=True,  # Enable sampling to allow more creative output
            temperature=0.7,  # Allow a bit more creativity and variation
            top_k=50,  # Limit the number of top options
            top_p=0.9,  # Use nucleus sampling for better coherence
            no_repeat_ngram_size=2,  # Avoid repeating n-grams in the text
            pad_token_id=tokenizer.eos_token_id  # Padding token to handle tokenization
        )
        
        # Decode the output tokens to text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the initial prompt to avoid repetition in the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Clean the generated text to ensure it stays on-topic
        if topic.lower() not in generated_text.lower():
            generated_text = f"Sorry, the content does not match the topic '{topic}'."

        # Ensure the word count is respected
        word_list = generated_text.split()
        generated_word_count = len(word_list)

        # If content is too short, extend it
        while generated_word_count < word_count:
            extra_content = model.generate(
                inputs["input_ids"],
                max_length=min((word_count * 5) + 200, 1024),  # Allow for more tokens
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
            additional_text = tokenizer.decode(extra_content[0], skip_special_tokens=True)
            generated_text += " " + additional_text
            word_list = generated_text.split()
            generated_word_count = len(word_list)

        # Trim to the exact word count if necessary
        generated_text = " ".join(word_list[:word_count])

        return generated_text

    except Exception as e:
        st.error(f"Error generating content: {e}")
        return "Sorry, I couldn't generate personalized content at this time."

# Main Streamlit interface
def main():
    st.title("Blog Content Generator")

    topic = st.text_input("Enter Blog Topic:")  # User can input any topic here
    word_count = st.number_input("Enter desired word count:", min_value=50, max_value=10000, value=500)
    target_audience = st.selectbox(
        "Select Target Audience:",
        ["general readers", "professionals", "students", "researchers", "content creators"]
    )

    if topic:
        if st.button("Generate Blog"): 
            blog_content = generate_blog_content(topic, word_count, target_audience)
            if blog_content:
                st.subheader("Generated Blog Post:")
                st.write(blog_content)
                st.write(f"Word Count: {len(blog_content.split())}")
            else:
                st.warning("Blog content was not generated.")
    else:
        st.warning("Please enter a topic to generate a blog.")

if __name__ == "__main__":
    main()
