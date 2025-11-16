from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import dotenv

dotenv.load_dotenv()

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
    You are SadakAI, responding as an experienced road-safety engineer who works with IRC, WHO, and global best-practice guidelines.
    Study the issue described and recommend interventions strictly based on the retrieved context. 
    Explain your reasoning the way a field expert would—practical, clear, and grounded.

    --------------------------------
    Retrieved Knowledge:
    {retrieved_docs}

    Road Issue:
    {road_issue}
    --------------------------------

    Write your response in a natural, expert tone:

    1. Begin with a short interpretation of the problem in your own words. 
      Explain what kind of issue this appears to be (visibility, speed, control, geometry, markings, etc.).

    2. Recommend interventions taken directly from the retrieved context.
      For each intervention:
      - Give it a meaningful title.
      - Explain why it fits this situation using real engineering judgment.
      - If the retrieved context includes a reference, mention it naturally in *italic* and bold like: *as referenced in IRC:35-2015, Clause 6.2*.

    3. Describe the expected safety impact—how these measures would realistically reduce conflicts or crash risk.

    4. Add any brief considerations or notes an engineer might include.

    Keep the tone confident, human, and grounded—never robotic or template-like. Avoid inventing interventions or sources outside the retrieved context.
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