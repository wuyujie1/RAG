import camelot
import fitz
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from parse_table import parse_table
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
import sys
from langchain.chains import LLMChain
from accelerate import Accelerator
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
import json
from preprocess_page import preprocess_for_knowledge_base


def process_pdf(file_path):
    context = ""
    with fitz.open(file_path) as doc:
        for index, page in enumerate(doc):
            context += "\n[PAGE]\n"
            tables = camelot.read_pdf(file_path, pages=str(index + 1), flag_size=True)
            last_y = 0
            sorted_tables = sorted(tables, key=lambda t: t._bbox[1], reverse=True)
            for j in range(tables.n):
                x0, y0, x1, y1 = sorted_tables[j]._bbox
                y0, y1 = page.rect.height - y1, page.rect.height - y0

                clip_rect = fitz.Rect(0, last_y, page.rect.width, y0)
                context += preprocess_for_knowledge_base(page.get_textbox(clip_rect)) + "\n"

                print(f"Start processing page {index + 1} table {j + 1}")
                if sorted_tables[j].parsing_report['whitespace'] == 100:
                    continue
                context += "\n<TABLE>\n"
                context += str(parse_table(sorted_tables[j].df, index + 1, j + 1, "./document/parsed_table"))
                context += "\n</TABLE>\n"
                last_y = y1
            # cut the footer
            footer_height = 54
            clip_rect = fitz.Rect(0, last_y, page.rect.width, page.rect.height - footer_height)
            context += preprocess_for_knowledge_base(page.get_textbox(clip_rect))
            context += "\n[/PAGE]\n"
    with open("context.txt", 'w', encoding='utf-8') as file:
        file.write(context)

    print(len(context))
    return context

def process_pdf_embedding(file_path):
    pages = []
    with fitz.open(file_path) as doc:
        for index, page in enumerate(doc):
            context = ""
            tables = camelot.read_pdf(file_path, pages=str(index + 1), flag_size=True)
            last_y = 0
            sorted_tables = sorted(tables, key=lambda t: t._bbox[1], reverse=True)
            for j in range(tables.n):
                x0, y0, x1, y1 = sorted_tables[j]._bbox
                y0, y1 = page.rect.height - y1, page.rect.height - y0

                clip_rect = fitz.Rect(0, last_y, page.rect.width, y0)
                context += preprocess_for_knowledge_base(page.get_textbox(clip_rect)) + "\n"

                print(f"Start processing page {index + 1} table {j + 1}")
                if sorted_tables[j].parsing_report['whitespace'] == 100:
                    continue
                context += "\n<TABLE>\n"
                context += str(parse_table(sorted_tables[j].df, index + 1, j + 1, "./document/parsed_table"))
                context += "\n</TABLE>\n"
                last_y = y1
            # cut the footer
            footer_height = 54
            clip_rect = fitz.Rect(0, last_y, page.rect.width, page.rect.height - footer_height)
            context += preprocess_for_knowledge_base(page.get_textbox(clip_rect))
            pages.append(Document(page_content=context, metadata={"source": "local"}))
            # context += "\n[/PAGE]\n"
    save_docs_to_jsonl(pages, 'context_embed.jsonl')
    return pages


def save_docs_to_jsonl(array, file_path:str):
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path):
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

if __name__ == "__main__":
    #documents = process_pdf_embedding("./document/EBTS v11.0_Final_508.pdf")
    # documents = open("context.txt", "r", encoding="utf-8").read()
    documents = load_docs_from_jsonl('context_embed.jsonl')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=25000,
        chunk_overlap=1000)

    text_chunks = text_splitter.split_documents(documents)

    # **Step 3: Load the Embedding Model***

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})

    # **Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    ##**Step 5: Find the Top 3 Answers for the Query***

    query = "What is TOT"
    docs = vector_store.similarity_search(query)

    # print(docs)
    llm = CTransformers(model="W:\codellama\codellama-34b-instruct.Q4_K_M.gguf",
                        model_type="llama",
                        config={'context_length': 100000,
                                'temperature': 0.01,
                                'max_new_tokens': 10000})

    template = """<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible
    using the context text provided. Your answers should only answer the question once and not have any text after
    the answer is done. If a question does not make any sense, or is not factually coherent, explain why instead of
    answering something not correct. If you don't know the answer to a question, please don't share false
    information.<</SYS>>

    [INST]
    Context:{context}
    \n\n
    Question:{question}
    [/INST]
    """

    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    # start=timeit.default_timer()

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': qa_prompt})

    # response=chain({'query': "YOLOv7 is trained on which dataset"})

    # end=timeit.default_timer()
    # print(f"Here is the complete Response: {response}")

    # print(f"Here is the final answer: {response['result']}")

    # print(f"Time to generate response: {end-start}")

    while True:
        user_input = input(f"prompt:")
        if query == 'exit':
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = chain({'query': user_input})
        print(f"Answer:{result['result']}")