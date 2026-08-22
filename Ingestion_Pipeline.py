import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# loading the documents using directory (document path as argument)
def load_documents(docs_path=""):
    if not os.path.exists(docs_path):
        raise FileNotFoundError (f"Directory {docs_path} doesn't exist")

    # a loader object which uses Directory loader
    # glob basically find files using username 
    # loader_cls specifies which loader to be used (using text files so TextLoader)
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    # checking if docs directory is not empty
    if len(documents)==0:
        raise FileNotFoundError (f"Directory {docs_path} doesn't contain any .txt files.")

    # printing basic info of text files in after they are loaded
    # Example:
    # documents = [
    #      Document( page_content = "Blah Blah Blah all text inside loaded text file",
    #                metadata = {'source': 'docs/nvidia.txt'})
    # ]

    # for i, DOCS in enumerate(documents):
    #     print(f"\n\nDocument{i+1}")
    #     print(f"Length: {len(DOCS.page_content)} Characters")
    #     print(f"Source: {DOCS.metadata['source']}")
    #     print(f"Content Preview : {DOCS.page_content[:50]}")
    #     print(f"MetaData: {DOCS.metadata}")

    return documents

def chunk_documents(document,chunk_size=800,chunk_overlap=50):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks= text_splitter.split_documents(document)

    # for i, Chunks in enumerate(chunks[:5]):
    #     print(f"\n\nChunks {i+1}")
    #     print(f"Length: {len(Chunks.page_content)} Characters")
    #     print(f"Source: {Chunks.metadata['source']}")
    #     print(f"Content: {Chunks.page_content}")

    # if len(chunks) > 5:
    #     print(f"and {len(chunks)-5} more chunks")
    
    return chunks

# Attempt to fix upserting by myself chunkID looks like {ID : {source : chunk_index}}
def calculate_chunk_IDs(chunks):

    prev_source = None
    chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata['source']

        if source == prev_source:
            chunk_index += 1

        else:
            prev_source=source
            chunk_index = 0

        chunkID = f"{source} : {chunk_index}"
        chunk.metadata['ID']= chunkID
    return chunks

def create_vector_db(chunks,chunk_IDs,db_path):
    print(f"{'-'*10}Creating Vector DB{'-'*10}")
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Embedding {len(chunks)} to {db_path}")

    vector_db= Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path,
        ids=chunk_IDs,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print(f"{'-'*10}Completed Vector DB{'-'*10}")

    return vector_db


def append_vector_db(chunks,chunk_IDs,db_path):
    print("Database Found. Preparing to Append")

    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )
    existing_items = vector_db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    new_ids = []
    for chunk, chunk_id in zip(chunks, chunk_IDs):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if len(new_chunks) > 0:
        print("Adding NEW chunks to the database")
        vector_db.add_documents(documents=new_chunks, ids=new_ids)
    else:
        print("No new documents found. The database is already up to date!")

def main():
    documents= load_documents(docs_path="docs")
    chunks=chunk_documents(documents)
    chunks= calculate_chunk_IDs(chunks) # these chunks contain ID

    chunk_IDs=[] # It will contain all the chunk IDs generated so far from new docs

    for i in chunks:
        ID_extract = i.metadata["ID"]
        chunk_IDs.append(ID_extract)

        db_path="db/chroma.db"
    if not os.path.exists(db_path):
        vector_DB=create_vector_db(chunks,chunk_IDs,db_path)
    else:
        vector_DB=append_vector_db(chunks,chunk_IDs,db_path)

if __name__ == "__main__":
    main()