import camelot
import fitz
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from parse_table import parse_table
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
)

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
                context += "<TABLE>\n"
                context += str(parse_table(sorted_tables[j].df, index + 1, j + 1, "./document/parsed_table"))
                context += "\n</TABLE>"
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
                context += "<TABLE>\n"
                context += str(parse_table(sorted_tables[j].df, index + 1, j + 1, "./document/parsed_table"))
                context += "\n</TABLE>"
                last_y = y1
            # cut the footer
            footer_height = 54
            clip_rect = fitz.Rect(0, last_y, page.rect.width, page.rect.height - footer_height)
            context += preprocess_for_knowledge_base(page.get_textbox(clip_rect))
            pages.append(Document(page_content=context, metadata={"source": "local"}))
            # context += "\n[/PAGE]\n"

    return pages



if __name__ == "__main__":
    # Read the images
    documents_images = SimpleDirectoryReader("./llama2/").load_data()

    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_index")

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # Create the MultiModal index
    index = MultiModalVectorStoreIndex.from_documents(
        documents_images,
        storage_context=storage_context,
    )

    retriever_engine = index.as_retriever(image_similarity_top_k=2)