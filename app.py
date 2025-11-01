from flask import Flask, render_template, request
from src.helper import download_embeddings,hyde_query_expansion,rerank_docs
from evaluation.eval import evaluate_with_threshold
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from src.prompt import *
import os
load_dotenv()
os.environ["PINECONE_API_KEY"] =os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

chatModel = ChatOpenAI(model="gpt-4o")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="gpt-4o-mini")


embedding = download_embeddings()
index_name="medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    pseudo_answer=hyde_query_expansion(input,llm)
    retrieved_docs = retriever.invoke(pseudo_answer)
    top_docs = rerank_docs(input, retrieved_docs,reranker)

    final_context="\n\n".join([d.page_content for d in top_docs])
    rag_chain = (
        RunnableParallel(
            {
                "context": lambda _: final_context,
                "input": RunnablePassthrough(),
            }
        )
        | prompt
        | chatModel
    )
    response = rag_chain.invoke({"input": msg})
    final_response,_=evaluate_with_threshold(msg,final_context,response.content)
    print("Final Response : ", final_response)
    return str(final_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)