""""
This script is modified from https://github.com/Optimistix/medical_QandA_assistant/blob/main/notebooks/evaluation-data-generation.ipynb
"""
from openai import OpenAI
from tqdm import tqdm
import json
from datasets import load_dataset
import pandas as pd


client = OpenAI()

# load the documents
df = pd.read_csv('data/medquad.csv')
df_sampled = df.sample(n=50)

documents = df_sampled.to_dict(orient='records')

prompt_template = """
You emulate a user of our medical question answering application.
Formulate 4 questions this user might ask based on a provided disease.
Make the questions specific to this disease.
The record should contain the answer to the questions, and the questions should
be complete and not too short. Use as few words as possible from the record. 

The record:

question: {question}
answer: {answer}
source: {source}
focus_area: {focus_area}

Provide the output in parsable JSON without using code blocks:

{{"questions": ["question1", "question2", ..., "question4"]}}
""".strip()


def generate_questions(doc):
    prompt = prompt_template.format(**doc)

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content
    return json_response

results = {}

for doc in tqdm(documents): 
    doc_id = doc['id']
    if doc_id in results:
        continue

    questions_raw = generate_questions(doc)
    questions = json.loads(questions_raw)
    results[doc_id] = questions['questions']

final_results = []

for doc_id, questions in results.items():
    for q in questions:
        final_results.append((doc_id, q))

df_results = pd.DataFrame(final_results, columns=['id', 'question'])
df_results.to_csv('data/ground-truth-retrieval.csv', index=False)