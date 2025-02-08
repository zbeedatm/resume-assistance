import streamlit as st
from dotenv import load_dotenv
from utils import *
from common import *
import os

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''
if 'app_candidates_relevant_docs' not in st.session_state:
    st.session_state['app_candidates_relevant_docs'] = None
if 'job_description' not in st.session_state:
    st.session_state['job_description'] = None
if 'document_count' not in st.session_state:
    st.session_state['document_count'] = None

def search_for_candidates(job_description, document_count):

    with st.spinner('Wait for it...'):
        #Create embeddings instance
        embeddings=create_embeddings_load_data()

        #Fecth relavant documents from PINECONE
        relevant_docs=similar_docs(job_description,
                                document_count if document_count else 1,
                                #os.environ['PINECONE_API_KEY'],
                                #"gcp-starter",
                                os.environ['PINECONE_HR_INDEX'],
                                embeddings,
                                None, #filter
                                #st.session_state['unique_id']
                                )

        #st.write(relavant_docs)

        # Store results in session state to persist across reruns
        st.session_state["app_candidates_relevant_docs"] = relevant_docs

        #Introducing a line separator
        # st.write(":heavy_minus_sign:" * 30)
        
        show_results(st.session_state["app_candidates_relevant_docs"],
                    "app_candidates_relevant_docs")

        if len(relevant_docs) > 0:
            st.success("Hope I was able to save your time ❤️")

        return relevant_docs
    
def main():
    load_dotenv()

    # st.sidebar.page_link("pagess/search.py", label="Search")
    # st.sidebar.page_link("pagess/admin.py", label="Administrator")

    st.set_page_config(page_title="Resume Screening Assistance")

    st.title("HR - Resume Screening Assistance 💁 ")
    #st.subheader("I can help you in resume screening process")

    candidates = []
    with st.form(key='search_form', border=1, clear_on_submit=False):    
    
        job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",
                                    # value=st.session_state['job_description'],
                                    key="job_description")
        document_count = st.text_input("No.of 'RESUMES' to return", 
                                    # value=st.session_state['document_count'],
                                    key="document_count")
        
        # if job_description:
        #     st.session_state['job_description'] = job_description
        # if document_count:
        #     st.session_state['document_count'] = document_count

        # submit=st.button("Find my candidates")
        submit=st.form_submit_button(label='Find my candidates')

    if submit and job_description and document_count:
        candidates = search_for_candidates(job_description, document_count)

    # if 'admin_candidates_relevant_docs' in st.session_state:
    if st.session_state["app_candidates_relevant_docs"] is not None:
        # print(st.session_state.get("app_candidates_relevant_docs"))
        show_results(st.session_state["app_candidates_relevant_docs"],
                     "app_candidates_relevant_docs")

    # Store results in session state to persist across reruns
    st.session_state["app_candidates_relevant_docs"] = candidates      


#Invoking main function
if __name__ == '__main__':
    from huggingface_hub import whoami

    try:
        print(whoami())  # This should print your Hugging Face account details
    except Exception as e:
        print("Invalid token or authentication issue:", e)

    main()
