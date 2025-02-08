import streamlit as st
from utils import *

def show_results(docs, relevant_docs_session_state_name):
    #Introducing a line separator
    # st.write(":heavy_minus_sign:" * 30)

    # with st.container(border=1):
    # #For each item in relevant docs - we are displaying some info of it on the UI
    # for item in range(len(relevant_docs)):
    for idx, (doc, score) in enumerate(docs):
        
        st.subheader("👉 " + str(idx+1))

        col1, col2 = st.columns([4, 1])
        with col1:
            #Displaying Filepath
            st.write("**File** : " + doc.metadata['name'])

        with col2:
            # Use session state to track button clicks
            if st.button(f"Remove", key=f'but_remove_{idx}'):
            # if st.form_submit_button(label='Remove'):
                remove_from_pinecone_by_file_id(os.environ['PINECONE_HR_INDEX'], [doc.metadata['file_id']])
                # remove_from_pinecone(os.environ['PINECONE_HR_INDEX'], [doc['vector_id']])
                
                # st.success("Candidate was removed successfully.")

                #del st.session_state[relevant_docs_session_state_name]  # Clear results after removal
                # Remove the deleted document from session state
                st.session_state[relevant_docs_session_state_name] = [
                    (d, s) for d, s in st.session_state[relevant_docs_session_state_name] if d.metadata['file_id'] != doc.metadata['file_id']
                ]
                st.rerun()  # Force a rerun to refresh results

        #Introducing Expander feature
        with st.expander('Show me 👀'): 
            # st.info("**Match Score** : " + str(relevant_docs[item][1]))
            #st.write("***"+relevant_docs[item][0].page_content)

            st.info("**Match Score** : " + str(score))
            
            #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
            # summary = get_summary(relevant_docs[item][0])
            summary = get_summary(doc)
            st.write("**Summary** : " + summary)

    st.session_state[relevant_docs_session_state_name] = None
