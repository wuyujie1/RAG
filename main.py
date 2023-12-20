from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
import sys
import json


MODEL_PATH = "codellama-34b-instruct.Q4_K_M.gguf"


def load_docs_from_jsonl(file_path):
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


if __name__ == "__main__":
    documents = load_docs_from_jsonl('context_embed.jsonl')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000)

    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en',
                                       model_kwargs={'device': 'cuda'})

    vector_store = FAISS.from_documents(text_chunks, embeddings)
    query = "List all the Type-2 subfields for TOT CAR and show min/max occurrence. For example: 2.001 LEN, min occurrence=1, max occurrence=1"
    docs = vector_store.similarity_search(query)
    print(docs)

    llm = CTransformers(model=MODEL_PATH,
                        model_type="llama",
                        config={'context_length': 40000,
                                'temperature': 0.01,
                                'max_new_tokens': 20000})

    template = """<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible
    using the context text provided. Your answers should only answer the question once and not have any text after
    the answer is done. If a question does not make any sense, or is not factually coherent, explain why instead of
    answering something not correct. If you don't know the answer to a question, please don't share false
    information.<</SYS>>

    [INST]
    Context:{context}
    Question:{question}
    [/INST]
    """
    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vector_store.as_retriever(search_kwargs={'k': 4}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': qa_prompt})

    while True:
        user_input = input(f"prompt:")
        if user_input == 'exit':
            print('Exiting')
            sys.exit()
        if user_input == '':
            continue
        result = chain({'query': user_input})
        print(f"Answer:{result['result']}")
        