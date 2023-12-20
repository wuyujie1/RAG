import camelot
import fitz
from parse_table import parse_table
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
import sys
import json
from preprocess_page import preprocess_for_knowledge_base
import pandas as pd


def get_table_header(context):
    reversed_context_strip_last_newline = context[::-1][1:]
    header_start = reversed_context_strip_last_newline.index("\n")
    header = context[len(context) - header_start-1:-1].strip(" ")
    return header


def process_pdf(file_path, embedding=False):
    pages = []
    contexts = ""
    with fitz.open(file_path) as doc:
        context = ""
        prev_table_header = ""
        prev_table = ""
        prev_table_start_index = -1
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
                curr_table_header = get_table_header(context)

                # new table
                if curr_table_header != prev_table_header:
                    # write last table to context
                    if str(prev_table) != "":
                        context = context[:prev_table_start_index] + "\n<TABLE>\n" + str(parse_table(prev_table, index + 1, j + 1, "./document/parsed_table")) + "\n</TABLE>\n" + context[prev_table_start_index:]

                    # update prev_table and header to current
                    prev_table_header = curr_table_header
                    prev_table = sorted_tables[j].df
                    prev_table_start_index = len(context)
                # same table across multiple pages
                else:
                    # first table
                    if str(prev_table) == "":
                        prev_table = sorted_tables[j].df
                    # now they are all pd dataframes
                    else:
                        current_table = sorted_tables[j].df

                        # Identifying common columns and rows
                        common_columns = prev_table.columns.intersection(current_table.columns)
                        common_rows = pd.merge(prev_table, current_table, how='inner')

                        # overlapping columns
                        if prev_table.iloc[0].equals(current_table.iloc[0]):
                            prev_table = pd.merge(prev_table, current_table, on=common_columns.tolist(), how='outer')
                        # columns not overlapping: treat as a new table
                        else:
                            if str(prev_table) != "":
                                context = context[:prev_table_start_index] + "\n<TABLE>\n" + str(
                                    parse_table(prev_table, index + 1, j + 1,
                                                "./document/parsed_table")) + "\n</TABLE>\n" + context[
                                                                                               prev_table_start_index:]

                            # update prev_table and header to current
                            prev_table_header = curr_table_header
                            prev_table = sorted_tables[j].df
                            prev_table_start_index = len(context)

                last_y = y1

            # cut the footer
            footer_height = 54
            clip_rect = fitz.Rect(0, last_y, page.rect.width, page.rect.height - footer_height)
            context += preprocess_for_knowledge_base(page.get_textbox(clip_rect))
            context += "\n[/PAGE]\n"
        if str(prev_table) != "":
            context = context[:prev_table_start_index] + "\n<TABLE>\n" + str(
                parse_table(prev_table, -1, -1, "./document/parsed_table")) + "\n</TABLE>\n" + context[prev_table_start_index:]
        contexts += context
        pages.append(Document(page_content=context, metadata={"source": "local"}))
    with open("context.txt", 'w', encoding='utf-8') as file:
        file.write(context)

    save_docs_to_jsonl(pages, "context_embed.jsonl")
    print(len(context))
    return pages if embedding else context


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
    # documents = process_pdf("./document/test0.pdf", True)
    documents = load_docs_from_jsonl('context_embed.jsonl')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000,
        chunk_overlap=1000)

    text_chunks = text_splitter.split_documents(documents)


    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device': 'cuda'})
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    #                                    model_kwargs={'device': 'cuda'})

    vector_store = FAISS.from_documents(text_chunks, embeddings)
    query = "List all the Type-2 subfields for TOT CAR and show min/max occurrence. For example: 2.001 LEN, min occurrence=1, max occurrence=1"
    docs = vector_store.similarity_search(query)
    print(docs)

    llm = CTransformers(model="W:\codellama\codellama-34b-instruct.Q4_K_M.gguf",
                        model_type="llama",
                        config={'context_length': 45000,
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
                                        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
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