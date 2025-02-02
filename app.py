#implement the embedding and rag pipeline now 
import chromadb, os, json, tiktoken, openai, gradio as gr
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()

openai_client = openai.OpenAI()

client = chromadb.PersistentClient(path="db")

import chromadb.utils.embedding_functions as embedding_functions

class ChainOfThought(BaseModel):
    reasoning: str
    answer: str
    
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small"
        )

collection = client.get_collection("company_docs", embedding_function=openai_ef)

#add the docs to the collection first and then we can do retrieval 
def add_documents_to_collection():
    #get the docs from the database 
    #clear the collection first 
    client.delete_collection("company_docs")

    collection = client.get_or_create_collection("company_docs", embedding_function=openai_ef)

    with open('company.json', 'r') as file:
        data = json.load(file)
    
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    max_tokens = 4000 #chunk size is 4000 tokens 

    for doc in data:
        markdown = data[doc]['markdown']
        full_text = f"{markdown}"
        tokens = encoding.encode(full_text)
        
        if len(tokens) > max_tokens:
            num_chunks = (len(tokens) + max_tokens - 1) // max_tokens  # Calculate the number of chunks
            chunk_size = (len(tokens) + num_chunks - 1) // num_chunks  # Calculate the size of each chunk
            chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            for i, chunk in enumerate(chunks):
                chunk_text = encoding.decode(chunk)
                print(f"Adding {doc}_part{i}, token count {len(chunk)}")
                collection.add(
                    documents = [chunk_text],
                    ids = [f"{doc}_part{i}"], #use the title and part number as the id
                    metadatas = [{"url": doc}]
                )
        else:
            print(f"Adding {doc}, token count {len(tokens)}")
            collection.add(
                documents = [full_text],
                ids = [doc], #use the title as the id
                metadatas = [{"url": doc}]
            )

user_prompt = """
You are <Company-Name> Support Assistant, created by Tautik. Your goal is to provide accurate, helpful responses to user queries about <Company-Name> by following this structured approach:


1. ANALYSIS
- Carefully analyze the user's query
- Identify the key issues or questions being asked
- Break down any technical or <Company-Name>-specific concepts

2. REASONING
- Review all provided help center articles thoroughly
- If exact answer isn't found, identify relevant contextual information
- Connect related information to form a complete answer
- Consider any documentation updates
- Explain your thought process for transparency

3. RESPONSE FORMULATION
- If exact answer exists: Provide direct answer with citations
- If exact answer doesn't exist: 
  * Synthesize information from related context
  * Explain implications and connections
  * Be clear about what is implied vs explicitly stated
- Always cite sources using [check here](url) format
- If information is insufficient, explain what's missing

IMPORTANT: DONT PUT THE ABOVE ANALYSIS, REASONING AND RESPONSE FORMULATION IN THE response. USE IT INTERALLY FOR YOUR REASONING.

Here are example responses:

Example 1:
Query: 
Reasoning: 
Answer: 

Example 2:
Query: 
Reasoning: 
Answer: 

Help Center Articles:
{context}

Documentation Updates:
{updates}

User Query:
{query}

Please provide your response following the above format and guidelines, ensuring to synthesize information when exact answers aren't available.
"""

# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are <Company-Name> Support Assistant, created by Tautik. Your goal is to provide accurate, helpful responses to user queries about <Company-Name> by following this structured approach:"),
     ("user", "{user_message}")]
)

output_parser = StrOutputParser()
llm = Ollama(model="deepseek-r1")

def get_relevant_context(query):
    #get the relevant context based on the query
    collection = client.get_collection("company_docs", embedding_function=openai_ef)
    results = collection.query(query_texts=[query], n_results=5)
    return results

def chatbot(query, chat_history):
    results = get_relevant_context(query)

    final_docs = set()
    with open('company.json', 'r') as file:
        data = json.load(file)
    
    for doc in results['metadatas'][0]:
        final_docs.add(f"{doc['url']}: {data[doc['url']]['markdown']}")

    #print the URLs so that we know what is going to the LLM 
    print("--------------------------------")
    print("Results: ", results)
    print(f"For Query: {query}, the URLs are: {results['metadatas'][0]}")
    print("--------------------------------")
    final_text = "\n".join(final_docs)
    with open('updated_docs.txt', 'r') as file:
        updates = file.read()   
    user_message = user_prompt.format(context=final_text, updates=updates, query=query)

    messages = []

    for exchange in chat_history:
        messages.append({"role": 'user', "content": exchange[0]})
        messages.append({"role": 'assistant', "content": exchange[1]})

    messages.append({"role": "user", "content": user_message})

    # Create the chain
    chain = prompt | llm | output_parser

    # Invoke the chain
    response_content = chain.invoke({"user_message": user_message})

    print("Response: ", response_content)

    try:
        # Parse the JSON string into a Python dictionary
        parsed_content = json.loads(response_content)
        reasoning = parsed_content.get('reasoning', '')
        answer = parsed_content.get('answer', '')
    except json.JSONDecodeError:
        # If parsing fails, treat the entire response as the answer
        reasoning = ''
        answer = response_content
    print("--------------------------------")
    print("Reasoning: ", reasoning)
    print("--------------------------------")
    return answer

def update_docs(updated_text):
    #we should also let the users update the docs as they need to 
    with open('updated_docs.txt', 'a') as file:
        #lets just write it to the updated text file for now. Pretty sure there are better ways to do this
        file.write(updated_text)

    return "Done", ""

if __name__ == "__main__":
    with gr.Blocks() as demo:
        chatbot_comp = gr.Chatbot(
            placeholder="<strong> <Company-Name></strong><br>Ask Me Anything", 
            height=600
        )
        gr.ChatInterface(
            fn = chatbot, 
            chatbot = chatbot_comp,
            title = "<Company-Name> Support Assistant", 
            examples = ["Sample Question 1?", 
                        "Sample Question 2",]
        )
        gr.Markdown("## Update Knowledge")
        gr.Interface(
            title = "Update Knowledge",
            fn = update_docs, 
            inputs = ["text"],
            outputs = ["text"]
        )
    demo.launch(share = True)
