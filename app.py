import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
# load the data from .env file with help of dotenv
from dotenv import load_dotenv

from time import sleep

 
load_dotenv()

# import the library for load the document
from langchain_community.document_loaders import PyPDFLoader

# import the library for split the text in the form of chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# import the library for google genrative ai api key and second one is for chat with google 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.vectorstores import InMemoryVectorStore

# connect to the llm gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


def document_process(path):

    # document loading from the local system
    loader = PyPDFLoader(path)
    docs = loader.load()

    # print(len(docs))
    # splitting 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200) #   chunk_size = 1000 means split the data into 1000 piceces and chunk_overlap means data is present in chunk_size and chunk_overla to built a relationship
    docs = splitter.split_documents(docs)
    # print(len(docs))

    # embadding and vectors stores
    embeddings = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-2-preview')
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

# user query
# query = 'In which university he studied?'

# document = vector_db.similarity_search(query=query, k=1)
# print(document)
# print(document[0])
# print(len(document), document[0].page_content)
# print(len(document[0].page_content))

# context = ""
# for doc in document:
#     context = context + doc.page_content + "\n\n"

# prompt = f"""
#     You are a helpful assistant and provide answer
#     based on the provided context. context:{context}, question: {query}
# """

st.subheader("Document A&A Chatbot - Ask Anything")

# check document uploaded or not if document not uploaded
# Initialize session state
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

# Document upload
if not st.session_state["document_uploaded"]:

    file = st.file_uploader(label="Select your PDF File", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
       
        with st.spinner("Processing....."):
            document_process("./uploaded_document.pdf")
        
        st.success("Document Uploaded Successfully.....")
        sleep(2)
        st.rerun()
        

if st.session_state["document_uploaded"] and st.session_state.vector_db:
    query = st.chat_input("Ask Anything.....")


    for oneMessage in st.session_state.messages:
        role = oneMessage['role']
        content = oneMessage['content']

        st.chat_message(role).markdown(content)
    # user query regarding to the document
    if query:

        # for store the user query 
        st.session_state.messages.append({'role':"user", "content":query})



        # used for showing the user query in the chat message
        st.chat_message("user").markdown(query)
        documents = st.session_state.vector_db.similarity_search(query, k=2)
        context =""

        for doc in documents:
            context = context+doc.page_content + "\n\n"

        prompt = f"""
        You are a helpful assistant and provide answer
        based on the provided context. context:{context}, question: {query}
        """

        # ai response
        with st.spinner(""):
            ai_response = llm.invoke(prompt)
            st.session_state.messages.append({'role':"ai", "content":ai_response.content[0]["text"]})
            st.chat_message("ai").markdown(ai_response.content[0]["text"])

# save the history of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
