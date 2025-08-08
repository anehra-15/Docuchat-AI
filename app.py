import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter # to split the text into chunks
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings # to convert text chunks into vector embeddings
import google.generativeai as genai # google sdk for interacting with GenAI models
from langchain.vectorstores import FAISS # to create a vectore store index of the vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() # for initializing env variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # for configuring the api key


#On the streamlit app, we had a left side wherein we will upload a pdf and on the right side will have a text bar wherein 
#we wil give the prompt to get our output
#function to read the pdf and get text from it
#This function takes multiple PDF documents, reads through every page of each PDF, extracts the text from every page, 
#concatenates all that text into one large string, and returns it.



def get_pdf_text(pdf_docs): # this function takes in a pdf doc
    text=""
    for docs in pdf_docs:
        pdf_reader=PdfReader(docs) # For each PDF document, it creates an instance of PdfReader (from PyPDF2), which allows reading the PDF's content and metadata.
        for page in pdf_reader.pages: # The function accesses the .pages attribute of the pdf_reader object, which is a list of all the pages in the PDF. It loops over each page.
            text+=page.extract_text()
    return text



#Function to split the text into chunks

#The overlap helps retain context between chunks so important information isn't lost at the boundaries.
#When you cut the book into chunks, sometimes important sentences or ideas start at the end of one chunk 
#and continue in the next chunk. If you don’t include any overlapping text, the next chunk might miss the 
#start of that idea — causing loss of context.

#By overlapping, you include some text from the end of one chunk again at the start of the next chunk, so 
#the transition is smooth and the model keeps the connection between chunks.
#This overlap helps keep continuity.

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


#Function to convert chunks to embeddings

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001") # this is a model provided by gogle to convert chunks into vector embeddings
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings) # take all this text chunks and embedd according to this embedding i have initialised
    vector_store.save_local("faiss_index") # we can store the vector embeddings in a vector db but since out input size is very small here so we are basicall savingthe vector embeddings in local here


#This function creates a question-answering chain using Google’s Gemini Pro chat model.

#It uses a prompt template that directs the model to answer questions based only on provided context.

#Temperature of 0.3 ensures somewhat predictable, reliable answers.

#Chain type "stuff" means the chain puts all context documents together as one chunk for the model to read
#before answering.

#Temperature ranges typically from 0 to 1 (sometimes higher).

#A low temperature (close to 0) makes the output very deterministic or predictable — the model tries to 
#give the most likely answer.

#A higher temperature (like 0.7 or above) makes the output more creative or varied — the model might 
#generate more diverse or imaginative answers.

#In your function, temperature=0.3 means the response will be relatively focused and less random, aiming 
#for clear, accurate answers.

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3) # Temperature is a parameter controlling the randomness or creativity of the model’s outputs.
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model, chain_type="stuff",prompt=prompt) # The "stuff" method simply concatenates ("stuffs") all the input texts together and sends them as one combined context prompt to the model.

    return chain



#Function to process the user input

#The function is designed to take a user question as input.

#It loads your FAISS embedding index (though the variable isn’t used afterwards—make sure you use it to 
#retrieve relevant docs if needed).

#It sets up the conversational QA chain.

#It asks the chain to answer the question using a context variable docs (likely your document chunks).

#Then prints and shows the AI's answer in the Streamlit UI


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True) # The embeddings object is passed to make sure the index uses the same embedding model for queries.
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs, "question":user_question}, return_only_outputs=True

    )
    print(response)
    st.write("Reply: ", response["output_text"]) # accessing the value associated with the key "output_text" in the dictionary called response.




#The user first uploads PDFs via the sidebar uploader.

#Then clicks Submit & Process to extract the PDFs' text, split it, embed it, and prepare the vector index.

#After that, the user can enter questions in the main page text box.

#Each question triggers a search & answer process using the vector store and Google Gemini chat model to 
#return an answer based on the PDF content.

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

