import streamlit as st
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import wikipedia
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Set up the LLMChain
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 64})
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app
def main():
    st.title("Wikipedia Search App")
    
    # User input
    user_input = st.text_input("Enter your search query:")
    
    if st.button("Search"):
        # Search Wikipedia
        try:
            page = wikipedia.page(user_input)
            content = page.content
            response = llm_chain.run(content, max_tokens=100)
            st.header(page.title)
            st.write(response)
        except wikipedia.exceptions.DisambiguationError as e:
            st.error("Multiple options found. Please refine your search query.")
        except wikipedia.exceptions.PageError:
            st.error("No results found. Please try a different search query.")

# Run the app
if __name__ == "__main__":
    main()
