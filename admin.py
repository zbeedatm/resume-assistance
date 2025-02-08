import streamlit as st
from utils import *
from common import *
import uuid

if 'admin_candidates_relevant_docs' not in st.session_state:
    st.session_state['admin_candidates_relevant_docs'] = None
if 'search_keywords' not in st.session_state:
    st.session_state['search_keywords'] = None


def search_for_candidates(search_by, keywords):
    
    with st.spinner('Wait for it...'):

        #Create embeddings instance
        embeddings=create_embeddings_load_data()

        if search_by == "File content":
            #Fecth relavant documents from PINECONE
            relevant_docs=similar_docs(keywords,
                                    5, #TODO
                                    os.environ['PINECONE_HR_INDEX'],
                                    embeddings,
                                    #None, #filter
                                    0.2, #TODO threshold
                                    )
        else:
            relevant_docs=similar_docs(keywords,
                                    10,
                                    os.environ['PINECONE_HR_INDEX'],
                                    embeddings,
                                    #{"name": {"$regex": keyword}} #Pinecone doesn't support regex
                                    #{"name": keyword} #I want to enable typing part of the file name
                                    )
            
            # Manually filter results where the metadata value contains the target substring
            relevant_docs = [
                (doc, score)
                for doc, score in relevant_docs
                if "name" in doc.metadata and keywords.lower() in str(doc.metadata["name"]).lower()
            ]

        # Store results in session state to persist across reruns
        # st.session_state["admin_candidates_relevant_docs"] = relevant_docs

        # Retrieve stored results (if available)
        # docs = st.session_state.get("relevant_docs", [])

        # if not docs:
        #     return

        show_results(relevant_docs, "admin_candidates_relevant_docs")

        return relevant_docs

def main():
    st.title("HR - Management 🗂️")

    tab_titles = ['Add', 'Remove']
    tabs = st.tabs(tab_titles)
    
    # Add content to each tab...
    with tabs[0]:
        # Upload the Resumes (pdf files)
        pdf = st.file_uploader("Upload resumes. Only PDF files allowed.", 
                               type=["pdf"],
                               accept_multiple_files=True)

        submit=st.button("Upload candidates")

        if submit:
            with st.spinner('Wait for it...'):
                st.write('**Note:** if a resume was existing before, then it will be skipped. No duplicates.')

                #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
                st.session_state['unique_id'] = uuid.uuid4().hex

                #Create a documents list out of all the user uploaded pdf files
                final_docs_list=create_docs(pdf, st.session_state['unique_id']) #TODO: unique_id is not in use

                #Displaying the count of resumes that have been uploaded
                st.write("*Resumes uploaded* :" + str(len(final_docs_list)))

                #Create embeddings instance
                embeddings=create_embeddings_load_data()

                #Push data to PINECONE
                push_to_pinecone(#os.environ['PINECONE_API_KEY'],
                                #"gcp-starter",
                                os.environ['PINECONE_HR_INDEX'],
                                embeddings,
                                final_docs_list)

                st.success("Candidates were uploaded ✔︎")
        
    with tabs[1]:
        # container = st.container()
        # with container:

        with st.form(key='remove_form', border=1, clear_on_submit=False):
            st.subheader("Search for candidates")

            keywords = st.text_input("Search for:" , 
                                    #  value=st.session_state['search_keywords'],
                                    #  key="search_keywords"
                                     )
            # if keywords:
            #     st.session_state['search_keywords'] = keywords

            search_by = st.radio(
                "Search by 🔎:",
                ["File name", "File content"],
                # key="something",
                # label_visibility=st.session_state.visibility,
                # disabled=st.session_state.disabled,
                horizontal=True,
            )

            # submit=st.button("Search")
            submit=st.form_submit_button(label='Search')

        candidates = []
        if submit and keywords:
            #st.header('Search for candidates to remove them from the database')
            candidates = search_for_candidates(search_by, keywords)

        # if 'admin_candidates_relevant_docs' in st.session_state:
        if st.session_state["admin_candidates_relevant_docs"] is not None:
            # print(st.session_state.get("relevant_docs"))
            show_results(st.session_state["admin_candidates_relevant_docs"], 
                        "admin_candidates_relevant_docs")
    
        # Store results in session state to persist across reruns
        st.session_state["admin_candidates_relevant_docs"] = candidates

#Invoking main function
if __name__ == '__main__':
    main()
