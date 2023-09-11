from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pickle
import os

load_dotenv()


class EmbeddingStore:
    def __init__(self):
        self.file_path = os.getenv("FILE_DIR")
        self.store_path = os.getenv("STORE_DIR")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30,   length_function=len
        )
        #all-MiniLM-L6-v2
        #self.model_name = "sentence-transformers/all-mpnet-base-v2"
        #self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.model_name = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"

        self.model_kwargs = {"device": "cuda"}

        self.embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        # self.embedding = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs)

    def all_embedding_vector(self, storeName: str, indexName: str):
        """
        디렉토리의 모든 파일을 임베딩한다.

        """
        pdf_documents = [
            os.path.join(self.file_path, filename)
            for filename in os.listdir(self.file_path)
        ]

        langchain_documents = []

        for document in pdf_documents:
            file_extension = get_file_extension(document)
            try:
                loader = self._get_loader(document, file_extension)
                data = loader.load()
                langchain_documents.extend(data)

            except Exception:
                continue

        print("Num pages: ", len(langchain_documents))
        print("Splitting all documents")

        split_docs = self.text_splitter.split_documents(documents=langchain_documents)

        print("Embed and create vector index")
        self._save_doc(indexName, split_docs)

    def embedding_vector(self, fileName: str, indexName: str):
        """
        파일 한건을 임베딩한다.
        """

        langchain_documents = []

        file_extension = get_file_extension(fileName)
        try:
            loader = self._get_loader(
                os.path.join(self.file_path, fileName), file_extension
            )

            print(os.path.join(self.file_path, fileName))
            data = loader.load()
            langchain_documents.extend(data)

        except Exception:
            print("파일을 찾을 수 없습니다.")

        split_docs = self.text_splitter.split_documents(documents=langchain_documents)

        print("Embed and create vector index")
        self._save_doc(indexName, split_docs)

    def _get_loader(self, document, file_extension):
        """
        파일종류에따른 로드를 반환한다
        """
        if file_extension == ".pdf":
            loader = PyPDFLoader(document)

        elif file_extension == ".csv":
            loader = CSVLoader(
                file_path=document,
                encoding="utf-8",
                csv_args={
                    "delimiter": ",",
                },
            )

        elif file_extension == ".txt":
            loader = TextLoader(file_path=document, encoding="utf-8")
        return loader

    def _save_doc(self, indexName, split_docs):
        """
        분할된 문서를 벡터저장소에 저장한다
        indexName : 한글제목은 저장안됨
        """

        indexName = indexName.replace(" ", "")  # 모든 공백 제거

        vectore_store = FAISS.from_documents(split_docs, embedding=self.embedding)

        path = os.path.join(self.store_path, indexName)

        vectore_store.save_local(path)


def get_file_extension(file_name) -> str:
    """
    파일 확장자를 리턴한다
    """
    file_extension = os.path.splitext(file_name)[1].lower()

    return file_extension


if __name__ == "__main__":
    print("open ai chat")
    em = EmbeddingStore()

    # em.all_embedding_vector("asd345g.pdf", "test01")
    #em.embedding_vector("asd345g.pdf", "test01")

    new_vectorstore = FAISS.load_local(
        "embedding/test01", em.embedding
    )
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(verbose=True, temperature=0), chain_type="stuff", retriever=new_vectorstore.as_retriever(),
    )

    res = qa.run("통합검색 도입에 따른 응용 SW 재개발 알려줘")
     
   
    print(res)
