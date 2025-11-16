from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import openai
import streamlit as st
import numpy as np

api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

try:
  loader = CSVLoader('context/GPT_Input_DB_clean.csv')
  docs = loader.load()
except Exception as e:
  print(f'Error: {e}')


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)


retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
# print(retriever)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = PromptTemplate(
    input_variables=["road_issue", "retrieved_docs"],
    template="""
You are SadakAI, responding as an experienced road-safety engineer who specializes in IRC, WHO, and global best-practice road-safety guidelines.

Before analyzing anything, check whether the user’s message is actually about a road-safety issue.

### If the message is NOT related to road safety:
Reply very briefly and politely, something like:
"Please describe the road safety issue you want help with."
Do not generate interventions, explanations, or long replies.

### If the message IS a road-safety issue:
Proceed with full expert analysis using the instructions below.

--------------------------------
Retrieved Knowledge:
{retrieved_docs}

Road Issue:
{road_issue}
--------------------------------

Write your response in a natural, expert human tone:

1. Begin with a short interpretation of the problem in your own words.  
   Identify the type of issue (visibility, speed, control, markings, conflict point, geometry, etc.).

2. Recommend interventions based strictly on the retrieved context.  
   For each intervention:
   - Provide a clear title.  
   - Explain why it fits, using practical engineering reasoning.  
   - If a reference appears in the retrieved context, mention it naturally in *italic*, e.g., *as noted in IRC:35-2015, Clause 6.2*.

3. Explain the expected safety impact—how the recommended measures reduce crash risk or improve road-user behavior.

4. Add any short notes or considerations an engineer might add (optional).

Keep the tone clear, expert, and grounded. Never invent interventions or references outside the retrieved context.
"""
)




def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'road_issue': RunnablePassthrough(),
    'retrieved_docs': retriever | RunnableLambda(format_docs)
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt_template | llm | parser
