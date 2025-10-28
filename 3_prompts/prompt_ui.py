from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv(dotenv_path = r'D:\langchain practice\.env')

hf_endpoint = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-R1',
    task = 'text-generation'
)
model = ChatHuggingFace(llm = hf_endpoint)

st.header('Research Tool')

paper_input = st.selectbox('Select Research Paper Name', ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox("Select Explanation Style", ['Beginner-Friendly', 'Technical', 'Code-Oriented', 'Mathmematical'])

length_input = st.selectbox("Select Explanation length", ['Short (1-2 paragraphs)', 'Medium (3-5 paragraphs)', 'Long (detailed explanation)'])

# template
template = PromptTemplate(
    template = 
"""Please summarize the research paper titled {paper_input} with the following
specifications:
Explanation style: {style_input}
Explanation length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies :-
    - Use relatable analogies to simply complex ideas.
    If certain analogies is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate and aligned with the provided style and length.

""",
input_variables=['paper_input', 'style_input', 'length_input']
)

# fill the placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})



if st.button('Summerize'):
    result = model.invoke(prompt)
    st.write(result.content)



"""Please summarize the research paper titled "(paper_input)" with the following
specifications:
Explanation style: {style_input}
Explanation length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies :-
    - Use relatable analogies to simply complex ideas.
    If certain analogies is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate and aligned with the provided style and length.

"""