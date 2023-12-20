from preprocess_page import preprocess_for_knowledge_base
import pandas as pd
import camelot
import fitz
from parse_table import parse_table
from langchain.docstore.document import Document


def get_table_header(context):
    reversed_context_strip_last_newline = context[::-1][1:]
    header_start = reversed_context_strip_last_newline.index("\n")
    header = context[len(context) - header_start-1:-1].strip(" ")
    return header


def save_docs_to_jsonl(array, file_path:str):
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')


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
