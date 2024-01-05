from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
import json

from accelerate import Accelerator

# for tiny size documents >>>
chunk_size = 10000
chunk_overlap = chunk_size * 0.1
k_vectors = 3
context_length = 33000
max_new_tokens = 20000
# <<<

MODEL_PATH = "W:\codellama\codellama-13b-instruct.Q4_K_M.gguf"


def load_docs_from_jsonl(file_path):
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


def find_all_subtypes(vectors, model):
    template = """<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible
    using the context text provided. Your answers should only answer the question once and not have any text after
    the answer is done.<</SYS>>

    [INST]
    Context:{context}
    Question:{question}
    [/INST]
"""
    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type='stuff',
                                        retriever=vectors.as_retriever(search_kwargs={'k': k_vectors}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': qa_prompt})

    prompt = """
    Find all logical record types presented in the tables. Output the type names, one name per row. For example:
    
type1
type2
...
    
    Note: Do not number the results in the list."""

    result = chain({'query': prompt})
    print(result['result'])
    with open('subtypes_dict.txt', 'w') as file:
        file.write(result['result'])
    return result['result']


def find_relation_for_subtype(vectors, model, subtype):
    template = """<<SYS>>You are a helpful, respectful, and honest assistant.
         Always answer as helpfully as possible using the context text provided ONLY, don't make up any answers.
          Your answers should only answer the question once and not have any text after the answer is done.<</SYS>>

    [INST]
    context: {context}
    prompt: {question}
    [/INST]
        """

#     prompt = f"""
#     Find all relations to the logical record type {subtype}.
#     List relationships in the format: [ENTITY 1, ENTITY TYPE, RELATION, ENTITY 2, ENTITY 2 TYPE].
#        - Do not number the results in the list.
#        - ENTITY TYPES are: "subtype" and "subfield".
#        - RELATIONSHIPS are defined as "has subfield" (linking a subfield to its subtype).
#     For example:
#
# ["AMN", "subtype", "has subfield", "2.001 LEN", "subfield"]
# ["AMN", "subtype", "has subfield", "2.002 IDC", "subfield"]
# ...
#
# """

    prompt = f"""
    For each logical record type presents in the contex:
    List its relationships in the format: [ENTITY 1, ENTITY TYPE, RELATION, ENTITY 2, ENTITY 2 TYPE].
       - Do not number the results in the list.
       - ENTITY TYPES are: "subtype" and "subfield".
       - RELATIONSHIPS are defined as "has subfield" (linking a subfield to its subtype).
    For example:

["AMN", "subtype", "has subfield", "2.001 LEN", "subfield"]
["AMN", "subtype", "has subfield", "2.002 IDC", "subfield"]
...

"""
    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type='stuff',
                                        retriever=vectors.as_retriever(search_kwargs={'k': k_vectors}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': qa_prompt})

    result = chain({'query': prompt})
    print(result['result'])
    return result['result']


if __name__ == "__main__":
    # process_pdf("./document/type2_table.pdf")
    documents = load_docs_from_jsonl('context_embed.jsonl')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    text_chunks = text_splitter.split_documents(documents)
    print(len(text_chunks))

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en',
                                       model_kwargs={'device': 'cuda'})

    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # GPU >
    accelerator = Accelerator()
    config = {'max_new_tokens': max_new_tokens, 'context_length': context_length, 'temperature': 0.01, 'gpu_layers': 41}
    # < GPU

    # CPU >
    # config = {'max_new_tokens': 100000, 'context_length': 100000, 'temperature':0.01}
    # < CPU

    llm = CTransformers(model=MODEL_PATH,
                        model_type="llama",
                        config=config)

    # GPU >
    llm, config = accelerator.prepare(llm, config)
    # < GPU

    # subtype_dict = find_all_subtypes(vector_store, llm)
    subtype_dict = [item.strip().strip("\n") for item in open("subtypes_dict.txt", "r").readlines()]

    relations = []
    for subtype in subtype_dict:
        raw_curr_relations = find_relation_for_subtype(vector_store, llm, subtype)
        with open('curr_relations.txt', 'w') as file:
            file.write(raw_curr_relations)
        cleaned_curr_relations = [item.strip().strip(",").strip("[").strip("]").split(",") for item in raw_curr_relations.split("\n")]
        formatted_json_list_curr = [{
            "source": item[0],
            "sourcetype": item[1],
            "relation": item[2],
            "target": item[3],
            "targettype": item[4]
        } for item in cleaned_curr_relations]
        relations.extend(formatted_json_list_curr)

    with open('./kg/kg-demo/src/kg_relations.json', 'w') as file:
        json.dump(relations, file, indent=4)
