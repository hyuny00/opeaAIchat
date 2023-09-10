import os

os.environ["OPENAI_API_KEY"] = "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"
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

if __name__ == "__main__":
    print("hi")
    pdf_path = "D:/Project/openAIChat/file/asd345g.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    #vectorstore = FAISS.from_documents(docs, embeddings)
    #vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(verbose=True, temperature=0), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("암호장비 노후화의 개선방안")
    print(res)
