from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract Data from the PDF File 
def load_pdf_file(data):
    loader = DirectoryLoader(
        data, 
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
 # Filter documents to retain only minimal necessary information.

 minimal_docs: List[Document] = [] 
 for doc in docs:
    src = doc.metadata.get("source")
    minimal_docs.append(
        Document(
            page_content=doc.page_content,
            metadata={"source": src}
        )
    )
 return minimal_docs

 def text_split(extracted_data):
     text_splitter = RecursiveCharacterTextSplitter(
        hunk_size=500,
        chunk_overlap=20,
     )
 texts_chunk = text_splitter.split_documents(extracted_data)
 return texts_chunk

# Download the embeddings from huggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    return embeddings