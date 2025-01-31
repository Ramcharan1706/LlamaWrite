import streamlit as st
from transformers import pipeline

def generate_blog_content(topic):
    generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    result = generator(f"Write a short blog about {topic}.", max_length=200, do_sample=True)
    return result[0]['generated_text']

def main():
    st.title("Simple AI Blog Generator")
    
    topic = st.text_input("Enter Blog Topic:")
    
    if st.button("Generate Blog"): 
        with st.spinner("Generating content..."):
            blog_content = generate_blog_content(topic)
            st.subheader("Generated Blog Post:")
            st.write(blog_content)

if __name__ == "__main__":
    main()
