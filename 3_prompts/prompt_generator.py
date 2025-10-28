# template
from langchain_core.prompts import PromptTemplate

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
input_variables=['paper_input', 'style_input', 'length_input'],
validate_template = True # to validate missing input variables
)

template.save('template.json')
