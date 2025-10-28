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

template = load_prompt(r'D:\langchain practice\template.json')

if st.button('Summerize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })

    st.write(result.content)