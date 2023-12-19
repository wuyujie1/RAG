import json
import pandas as pd

def parse_json_to_str(json_obj, indent):
    formatted_json = ""
    for key in json_obj:
        formatted_json += indent * " " + key + ":"
        if type(json_obj[key]) == dict:
            formatted_json += "{\n" + parse_json_to_str(json_obj[key], indent + 4) + "\n" + indent * " " + "},\n"
        else:
            formatted_json += " " + str(json_obj[key]) + ",\n"
    return formatted_json

def parse_table(table, page_num, table_num, out_dir):
    num_rows, num_cols = table.shape
    json_content = {}

    col_names = table.iloc[0]

    col_items = [table[i][table[i] != ""].count() for i in range(num_cols)]
    row_names_index = col_items.index(max(col_items))
    row_names = table[row_names_index]
    for i in range(0, num_cols):
        if i == row_names_index:
            continue
        json_content[table.iloc[0][i]] = {}
    for i in range(0, num_cols):
        last_non_empty_row_cell = table[i][1]
        for j in range(1, num_rows):
            if i == row_names_index:
                continue
            if table[i][j] != "" and row_names_index != 0:
                last_non_empty_row_cell = table[i][j]
            elif row_names_index == 0:
                last_non_empty_row_cell = table[i][j]
            entry_val = last_non_empty_row_cell
            if "<s>" in last_non_empty_row_cell:
                super_script_start_index = entry_val.index("<s>")
                target = entry_val[:super_script_start_index].strip() + ", check superscript reference " + entry_val[super_script_start_index:].strip("</s>").strip()
            else:
                target = last_non_empty_row_cell
            if col_names[i] in json_content and row_names[j] in json_content[col_names[i]]:
                if type(json_content[col_names[i]][row_names[j]]) == str:
                    json_content[col_names[i]][row_names[j]] = [json_content[col_names[i]][row_names[j]], target]
                else:
                    json_content[col_names[i]][row_names[j]].append(target)
            else:
                json_content[col_names[i]][row_names[j]] = target
    formatted_json = parse_json_to_str(json_content, 0)
    with open(f"{out_dir}/page_{page_num}_table_{table_num}.json", 'w') as f:
        json.dump(json_content, f, indent=4)
    return formatted_json

# def parse_table(table, page_num, table_num, out_dir):
#     num_rows, num_cols = table.shape
#     descriptive_text = ""
#
#     # Extracting column names, assuming first row contains headers
#     col_names = table.iloc[0]
#     last_known_primary_val = ""
#
#     for row_idx in range(1, num_rows):
#         primary_val = table.iloc[row_idx, 0] if pd.notna(table.iloc[row_idx, 0]) and table.iloc[row_idx, 0] != "" else last_known_primary_val
#         if primary_val:
#             last_known_primary_val = primary_val
#
#         row_description = f"{col_names[0]} '{primary_val}' contains: "
#         row_data = []
#
#         for col_idx in range(1, num_cols):
#             cell_value = table.iloc[row_idx, col_idx]
#             if pd.notna(cell_value) and cell_value != "":
#                 col_description = f"{col_names[col_idx]} as '{cell_value}'"
#                 row_data.append(col_description)
#         row_description += ", ".join(row_data) + ". " if row_data else ""
#         descriptive_text += row_description
#
#     return descriptive_text


# def parse_table(table, page_num, table_num, out_dir):
#     content = ""
#
#     col_names = table.iloc[0]
#     content += ",".join(col_names) + "\n"
#
#     last_non_empty_primary = None
#     for index, row in table.iterrows():
#         if index == 0:
#             continue
#         if row[0] != "":
#             last_non_empty_primary = row[0]
#         else:
#             row[0] = last_non_empty_primary
#         content += ",".join([str(value) for value in row]) + "\n"
#
#     return content
