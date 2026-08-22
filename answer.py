from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from Retrieval_Pipeline import get_vector_db, get_relevantChunks
from dotenv import load_dotenv

load_dotenv()

def get_LLM():
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        # task="text-generation",
        max_tokens= 1024,
        temperature = 0.3
    )
    return llm

def create_prompt(query,relevant_chunks):
    system_prompt= (f"""Based on the following documents answer the question provided

    Documents: 
    {chr(10).join(f"- {doc.page_content}" for doc in relevant_chunks)}

    Please provide a clear helpful answer using only the information from these documents. If you can't find an answer in the document, say " I don't have enought information to answer that question based on the provided documents.""")

    message= [
        SystemMessage(content = f"{system_prompt}")
    ] + chatHistory + [
        HumanMessage(content = f"{query}")
    ]
    return message

chatHistory = []
vector_db= get_vector_db()
llm = get_LLM()

def ask_question(query):

    if chatHistory:
        messages= [
            SystemMessage(content = "Given the chat history , rewrite the new question to be standalone and searchable. Just return the rewritten Question.")
            ] + chatHistory + [
            HumanMessage(content = f"New question: {query}")
        ]
        result = llm.invoke(messages)
        new_query= result.content.strip()

    else:
        new_query = query

    return new_query

def main():
    print("Main Function")

    while True:
        print("\n\nEnter quit to exit")
        query = str(input("\nEnter your question: "))

        if query.lower() == "quit":
            break
        
        else:
            new_query = ask_question(query)
            relevant_chunks=get_relevantChunks(vector_db,new_query)
            if not relevant_chunks:
                print("I don't have enough information to answer the question based on provided documents.")
                continue

            message = create_prompt(query,relevant_chunks)
            response = llm.invoke(message)
            print(f"{response.content}\n\n")

            chatHistory.append(HumanMessage(content= query))
            chatHistory.append(AIMessage(content = response.content))

            del chatHistory[:-6]


if __name__ == "__main__":
    main()