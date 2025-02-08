from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document

import os
import hashlib

#import pinecone
from pinecone import Pinecone as PineconeClient
#from langchain.vectorstores import Pinecone  #This import has been replaced by the below one :)
from langchain_community.vectorstores import Pinecone

from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub

from dotenv import load_dotenv
import time


if not os.environ.get("PINECONE_API_KEY"):
    load_dotenv()

pinecone_client = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
#pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Calculate hash code for the document
        hash_code=calculate_content_hash(chunks) 

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,
                    #   "file_id":filename.file_id, #TODO - the file_id was unique, but not any more!
                    "file_id":hash_code,
                    "type=":filename.type,
                    "size":filename.size,
                    "unique_id":unique_id, #TODO: Not in use. for different runs/sessions handling.
                    #   "hash_code":hash_code
                    },
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def filter_existing_docs(index_name, docs):
    # Initialize the Pinecone index
    index = pinecone_client.Index(index_name)

    # Extract hash_codes from the docs list using the appropriate method for your Document objects
    # hash_codes = [doc.metadata['hash_code'] for doc in docs]  # Accessing 'metadata' if it's an attribute
    # print("Hash Codes:", hash_codes)
    ids = [doc.metadata['file_id'] for doc in docs]  # Accessing 'metadata' if it's an attribute
    # print("ids:", ids)

    # Fetch by list of hash_codes (ensure hash_codes are valid ids)
    # fetch_response = index.fetch(ids=ids)  
    # print("Fetch Response:", fetch_response)

    # Get the existing hash_codes that are already in the Pinecone index
    # existing = set(fetch_response.get('vectors', {}).keys())  # Extract existing IDs from the response
    # print("1 -----------> Existing:", len(existing))

    # Search by metadata
    query_filter = {"file_id": {"$in": ids}}
    # search_response = index.query(filter=query_filter, top_k=len(ids))
    search_response = index.query(
        vector=[0] * 384,  # Use a placeholder vector with the same dimensionality
        filter=query_filter,
        top_k=len(ids) if len(ids)>0 else 1,
        include_metadata=True  # Ensure metadata is included in the response
    )
    existing_ids = [match['metadata']['file_id'] for match in search_response.get('matches', [])]
    #print("1 -----------> Existing:", existing_ids)

    # Filter out the docs that have already been added to Pinecone
    filtered_docs = [doc for doc in docs if doc.metadata['file_id'] not in existing_ids]
    # print("2 -----------> Filtered Docs:", len(filtered_docs))

    return filtered_docs


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(#pinecone_apikey,
                     #pinecone_environment,
                     pinecone_index_name,
                     embeddings,
                     docs):

    # pinecone.init(
    #     api_key=pinecone_apikey,
    #     environment=pinecone_environment
    # )
    
    # Filter out documents that already exist in the Pinecone index
    filtered_docs = filter_existing_docs(pinecone_index_name, docs)

    Pinecone.from_documents(filtered_docs, embeddings, index_name=pinecone_index_name)
    

#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(
        #pinecone_apikey,
        #pinecone_environment,
        pinecone_index_name,
        embeddings):
    # For some of the regions allocated in pinecone which are on free tier, the data takes upto 10secs for it to available for filtering
    #so I have introduced 20secs here, if its working for you without this delay, you can remove it :)
    #https://docs.pinecone.io/docs/starter-environment
    # print("20secs delay...")
    # time.sleep(20)

    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index


#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,
                 k,
                 pinecone_index_name,
                 embeddings,
                 #filter,
                 threshold=None
                 ):

    index = pull_from_pinecone(pinecone_index_name,
                               embeddings)
    #The unique_id is to retrieve the docs per each run, each run will have a different unique_id
    #But in my case, I didn't use it. 
    #similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    similar_docs = index.similarity_search_with_score(query, int(k)) #, filter=filter)
    
    # Filter results based on similarity score threshold
    if threshold:
        matches = [(doc, score) for doc, score in similar_docs if score >= threshold]
        return matches
    
    # print("-------------->", similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)

    # llm = HuggingFaceHub(#repo_id="bigscience/bloom", 
    #                     repo_id="samkeet/GPT_124M",
    #                     model_kwargs={"temperature":1e-10})

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
 
    return summary

    # from transformers import pipeline
    # summarizer = pipeline("summarization", model="google/flan-t5-large")
    # summary = summarizer(current_doc, max_length=150, min_length=50, do_sample=False)

    # return summary[0]['summary_text']

def get_vector_ids_by_file_id(index_name, file_id):
    index = pinecone_client.Index(index_name)

    # Query Pinecone with metadata filter
    query_results = index.query(
        vector=[0] * 384,  # Dummy vector (must match your embedding size)
        filter={"file_id": file_id},  # Exact-match filter
        top_k=10,  # Adjust as needed
        include_metadata=True
    )

    # Extract vector IDs
    vector_ids = [match["id"] for match in query_results["matches"]]

    return vector_ids


def remove_from_pinecone_by_file_id(index_name, file_ids):
    index = pinecone_client.Index(index_name)

    for file_id in file_ids:
        # Retrieve vector IDs using metadata filtering
        vector_ids = get_vector_ids_by_file_id(index_name, file_id)

        if vector_ids:
            # Delete vectors by their IDs
            index.delete(ids=vector_ids)
            print(f"Deleted vectors with file_id: {file_id} and vector_ids: {vector_ids}")
        else:
            print(f"No vectors found with file_id: {file_id}")


# def remove_from_pinecone(index_name, ids):
#     # Connect to your index
#     index = pinecone_client.Index(index_name)

#     # Delete the document by ID
#     index.delete(ids=ids)


# def remove_from_pinecone_by_file_id(index_name, file_ids):
#     # Connect to your Pinecone index
#     index = pinecone_client.Index(index_name)

#     # Iterate over each file_id provided
#     for file_id in file_ids:
#         # Perform a query to search for the vector(s) associated with the 'file_id' in metadata
#         query_results = index.query(
#             filter={"metadata.file_id": file_id},  # Search by file_id in metadata
#             top_k=1  # We're assuming there should be a single match
#         )
        
#         # Check if there are any results
#         if query_results["matches"]:
#             # Extract the vector IDs of the matched results
#             vector_ids = [match["id"] for match in query_results["matches"]]
            
#             # Delete the vectors based on their IDs
#             index.delete(ids=vector_ids)
#             print(f"Deleted vectors with file_id: {file_id} and ids: {vector_ids}")
#         else:
#             print(f"No vectors found with file_id: {file_id}")

def calculate_content_hash(content):
    # Calculate SHA-256 hash of the content
    return hashlib.sha256(content.encode()).hexdigest()


    