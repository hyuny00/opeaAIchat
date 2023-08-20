
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
import os

load_dotenv()



class EmbeddingStore:

    def __init__(self):
        self.file_path = os.getenv("FILE_DIR");
        self.store_path = os.getenv("STORE_DIR");


    
        self.text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=100,
                            length_function=len
        )

        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {"device": "cuda"}

        self.embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        #self.embedding = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs)

    

    def all_embedding_vector(self, storeName : str):
         
        pdf_documents = [os.path.join(self.file_path, filename) for filename in os.listdir(self.file_path)]

     
        langchain_documents = []

        for document in pdf_documents:
            file_extension=get_file_extension(document)
            try:
                if file_extension == ".pdf":
                    loader = PyPDFLoader(document)

                elif file_extension == ".csv":
                    loader = CSVLoader(file_path=document, encoding="utf-8",csv_args={ 'delimiter': ',',})

                elif file_extension == ".txt":
                    loader = TextLoader(file_path=document, encoding="utf-8")

                data = loader.load()   
                langchain_documents.extend(data)
                
            except Exception:
                continue

        print("Num pages: ", len(langchain_documents))
        print("Splitting all documents")

        split_docs = self.text_splitter.split_documents(documents=langchain_documents)


        print("Embed and create vector index")
        self.save_doc(storeName, split_docs)





    def save_doc(self, storeName, split_docs):
        vectore_store = FAISS.from_documents(split_docs, embedding=self.embedding)
        vectore_store.save_local( self.store_path ,index_name=storeName)


def get_file_extension(file_name):
    file_extension =  os.path.splitext(file_name)[1].lower()
            
    return file_extension
      
if __name__ == '__main__':
    print('open ai chat')
    em = EmbeddingStore()

    em.all_embedding_vector("test01")

    
