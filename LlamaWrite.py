import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set page configuration
st.set_page_config(page_title="Blog Content Generator")

# Ensure offload folder exists for large models
offload_folder = "./offload"
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

# Load Hugging Face models for text generation
@st.cache_resource(ttl=3600)  # Cache models for 1 hour
def load_models():
    try:
        # Load LLaMA 2 model (ensure you use the correct LLaMA 2 model from Hugging Face)
        model_name = "meta-llama/Llama-2-7b-hf"  # LLaMA 2 model (you can choose other variants like 13b)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        st.write(f"Model {model_name} loaded successfully.")
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Function to generate blog content using LLaMA 2
def generate_blog_content(topic, word_count, target_audience):
    model, tokenizer = load_models()

    if model is None:
        return "Sorry, I couldn't generate personalized content at this time."

    try:
        # Dynamically create a prompt based on the user's input parameters
        prompt = (
            f"Write a detailed blog post about '{topic}', tailored to {target_audience}. "
            f"The post should be engaging, informative, and relevant, covering aspects such as the history, culture, "
            f"landmarks, traditions, food, and places of interest related to '{topic}'. The tone should be professional, "
            f"but also accessible to {target_audience}. The content should be at least {word_count} words long and must focus "
            f"solely on '{topic}', avoiding unrelated information."
        )

        # Tokenize the prompt and generate content using sampling to reduce repetition
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Parameters to control generation: temperature for creativity, top_k for reducing repetition
        output = model.generate(
            inputs["input_ids"],
            max_length=word_count * 5,  # Multiply word count by 5 for token length
            num_return_sequences=1,  # Return a single sequence
            do_sample=True,  # Enable sampling to improve randomness
            temperature=0.7,  # Control randomness (lower = more deterministic)
            top_k=50,  # Limit the number of top options to sample from
            top_p=0.95,  # Nucleus sampling: cumulative probability threshold
            no_repeat_ngram_size=2,  # Prevent n-grams of size 2 from repeating
            pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled correctly
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean up the generated text (e.g., removing unwanted intro text)
        if "Write a" in generated_text:
            generated_text = generated_text.split("Write a", 1)[-1]

        # Ensure the content is as close to the desired word count as possible
        word_list = generated_text.split()
        generated_word_count = len(word_list)

        # If the content is too short, append more content until we reach the word count
        while generated_word_count < word_count:
            extra_content = model.generate(
                inputs["input_ids"],
                max_length=(word_count * 5) + 200,  # Allow more tokens to keep adding content
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
            additional_text = tokenizer.decode(extra_content[0], skip_special_tokens=True)
            generated_text += " " + additional_text
            word_list = generated_text.split()
            generated_word_count = len(word_list)

        # Cut the content to the exact word count
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
