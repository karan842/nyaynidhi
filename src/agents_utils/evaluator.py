from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from qdrant_client import QdrantClient
from src.agents_utils.retriever import vector_store
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv
from datasets import Dataset
import os
import json
import pandas as pd

from ragas import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Load config 
config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

# Collection name
collection_name = config['COLLECTION_NAME']['ipc-800']

## Connect to Qdrant client
client_url = QdrantClient(url="http://localhost:6333/")

# Models
llm = config['LLAMA']
embedding_model = config['EMBEDDING_MODEL']

def evaluate_result():
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=gemini_api_key)

    llm = ChatGroq(groq_api_key=groq_api_key,
                    model=llm,
                    temperature=0.2)

    # Define vector retriever
    vector_db = vector_store(embeddings, client_url, collection_name)
    retriever = vector_db.as_retriever(top_k=20)

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    PROMPT_TEMPLATE = """
        Go through the context and answer given question strictly based on context.
        Context: {context}
        Question: {question}
        Answer:
    """

    qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
    )

    queries = [
        "What is the IPC section for Rape case? Describe it in depth"
    ]

    ground_truths = [
        """A man is said to commit "rape" if he--
    (a) penetrates his penis, to any extent, into the vagina, mouth, urethra or anus of a woman or makes her to do so with him or any other person; or
    (b) inserts, to any extent, any object or a part of the body, not being the penis, into the vagina, the urethra or anus of a woman or makes her to do so with him or any other person; or
    (c) manipulates any part of the body of a woman so as to cause penetration into the vagina, urethra, anus or any part of body of such woman or makes her to do so with him or any other person; or
    (d) applies his mouth to the vagina, anus, urethra of a woman or makes her to do so with him or any other person,""",
    ]

    results = []
    contexts = []
    for query in queries:
        result = qa_chain({"query": query})

        results.append(result['result'])
        sources = result["source_documents"]
        contents = []
        for i in range(len(sources)):
            contents.append(sources[i].page_content)
        contexts.append(contents)

    d = {
        "question": queries,
        "answer": results,
        "contexts": contexts,
        "ground_truth": ground_truths
    }


    dataset = Dataset.from_dict(d)
    score = evaluate(dataset, llm=llm, embeddings=embeddings, metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness])
    score_df = score.to_pandas()
    score_df.to_json("./artifacts/EvaluationScores.json", orient="records", indent=4)
    print("\n\n\n")
    print(score_df[['faithfulness','answer_relevancy', 'context_precision', 'context_recall',
            'context_entity_recall', 'answer_similarity', 'answer_correctness',
            'harmfulness']].mean(axis=0))

if __name__ == '__main__':
    print("Performing evaluation")
    evaluate_result()


