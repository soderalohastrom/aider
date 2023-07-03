import os
import json
import itertools
import torch
import math 
import pinecone
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings

print(os.getenv('OPENAI_API_KEY'))
model_name = 'text-embedding-ada-002'

import os
os.environ['OPENAI_API_KEY'] = "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"
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
llm=OpenAI(model_name="text-embedding-ada-002")
print("Imported necessary libraries")

# init connection to pinecone
pinecone.init(
    api_key='ae7236b9-b523-4e96-a95d-b6aadd565e06',  # your pinecone.io API key
    environment='us-east-1-aws'  # your environment
)

print("Initialized connection to Pinecone")

index = pinecone.Index('kelleher-dolls')  # your index name

print("Initialized index")

# load the models
print("Loading models...")
model_id = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_id)
sparse_model = AutoModelForMaskedLM.from_pretrained(model_id)
print("Models loaded")

def generate_sparse_vectors(questions):
    print(f"Generating sparse vectors for {len(questions)} questions...")
    vectors = []
    for question in questions:
        inputs = tokenizer(question, return_tensors='pt')
        outputs = sparse_model(**inputs)[0]
        # Use mean pooling to get a fixed size vector
        vector = outputs.mean(dim=1).tolist()
        vectors.append(vector)
    print("Sparse vectors generated")
    return vectors

def hybrid_query(question, top_k, alpha):
    print(f"Performing hybrid query with question: {question}, top_k: {top_k}, alpha: {alpha}")
    # convert the question into a sparse vector
    sparse_vec = generate_sparse_vectors([question])
    # convert the question into a dense vector using OpenAI LLM
    dense_vec = llm.encode([question]).tolist()
    # set the query parameters to send to pinecone
    query = {
      "top_k": top_k,
      "vector": dense_vec,
      "sparse_vector": sparse_vec[0],
      "alpha": alpha,
      "include_metadata": True
    }
    print("Query parameters set")
    # query pinecone with the query parameters
    print("Sending query to Pinecone...")
    result = index.query(**query)
    print("Received results from Pinecone")
    # return search results as json
    return result

question = "who lives in Ohio?"
print(f"Question: {question}")
results = hybrid_query(question, top_k=3, alpha=1)
print("Results:")
for result in results:
    print(result)