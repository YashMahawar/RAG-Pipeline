from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def get_vector_db(db_path="db/chroma.db"):

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )
    return vector_db

def get_relevantChunks(vector_db,query):
    retriever=vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":5,
            "score_threshold" : 0.3
            }
        )
    relevant_chunks= retriever.invoke(query)
    return relevant_chunks

    
# def main():
#     print ("Main Function")
#     db_path="db/chroma.db"
#     vector_db = get_vector_db(db_path)

#     query = "What should I do if my robot stops moving and flashes an amber light?"

#     relevant_chunks= get_relevantChunks(vector_db,query)

#     # print(f"Query: {query}")
#     # print(f"{'-' * 10}Context{'-'*10}")
#     # for i, doc in enumerate (relevant_chunks):
#     #     print (f"Document {i}:\n{doc.page_content}\n")

# if __name__ == "__main__":
#     main()