def parse_pdf_header(pdf_content):
    # The PDF header starts with '%PDF-' and is usually within the first few lines
    header_position = pdf_content.find(b'%PDF-')
    if header_position == -1:
        raise ValueError("Not a valid PDF file")

    # Look for the first newline character to find the end of the header
    # Newline characters can be either '\n', '\r\n', or '\r'
    header_end = pdf_content.find(b'\n', header_position)
    if header_end == -1:
        header_end = pdf_content.find(b'\r\n', header_position)
    if header_end == -1:
        header_end = pdf_content.find(b'\r', header_position)

    # If no newline character is found, the header might be followed by a comment or object definition
    # In this case, limit the header to the first 20 bytes after '%PDF-'
    if header_end == -1:
        header_end = header_position + 20

    # Decode with 'ignore' to avoid UnicodeDecodeError
    return pdf_content[header_position:header_end].decode('utf-8', 'ignore')


def find_xref_position(pdf_content):
    # The xref position is typically near the end of the file
    xref_position = pdf_content.rfind(b'startxref')
    if xref_position == -1:
        raise ValueError("xref not found")

    # Skip the 'startxref' keyword and any subsequent whitespace
    xref_position_start = xref_position + len(b'startxref')
    while pdf_content[xref_position_start] in b'\x20\x0A\x0D\x09':  # Space, LF, CR, Tab
        xref_position_start += 1

    # Find the end of the xref position number
    xref_position_end = xref_position_start
    while pdf_content[xref_position_end] not in b'\x20\x0A\x0D\x09':  # Space, LF, CR, Tab
        xref_position_end += 1

    # Convert the xref position to an integer
    xref_position_number = int(pdf_content[xref_position_start:xref_position_end])

    return xref_position_number


def get_object(pdf_content, obj_num, xref_table):
    # This is a simplified example and does not handle all object types or errors
    obj_pos = xref_table[obj_num]
    obj_header = pdf_content[obj_pos:].split(b'\n')[0]
    if b'stream' in obj_header:
        # Handle stream object
        pass  # ...
    else:
        # Handle other object types
        pass  # ...


def split_pdf_objects(pdf_content, xref_table):
    object_dict = {}
    sorted_offsets = sorted(list(xref_table.values()))

    for i, obj_offset in enumerate(sorted_offsets):
        obj_pos = obj_offset
        if i + 1 < len(sorted_offsets):
            # If this is not the last known object, find the end position based on the next object's offset
            obj_end = sorted_offsets[i + 1]
        else:
            # If this is the last known object, find the 'endobj' keyword to determine the end of the object
            obj_end = pdf_content.find(b'endobj', obj_pos) + len('endobj')
            if obj_end == -1 + len('endobj'):  # if 'endobj' is not found
                obj_end = len(pdf_content)  # use the end of the pdf content

        # Extract objects between current object and next object or end of file
        between_objects_data = pdf_content[obj_pos:obj_end]

        for obj_start in find_substring_occurrences(b' obj', between_objects_data):
            # Check if there are two ints before this occurrence in the form of "i1 i2 obj"
            space_pos = between_objects_data.rfind(b' ', 0, obj_start)
            if space_pos == -1:
                continue  # Malformed object declaration, skip to next occurrence

            # Extract the substring that should contain "i1 i2"
            i1_i2_substring = between_objects_data[max(0, space_pos - 10):obj_start]
            parts = i1_i2_substring.split()
            if len(parts) < 2 or not (parts[-2].isdigit() and parts[-1].isdigit()):
                continue  # Malformed object declaration, skip to next occurrence

            # Find the next 'endobj' and save the entire string from including i1 and endobj as value
            end_pos = between_objects_data.find(b'endobj', obj_start) + len('endobj')
            if end_pos == -1 + len('endobj'):  # if 'endobj' is not found
                continue  # Malformed object declaration, skip to next occurrence

            # Take the 'i1 i2 obj' as a key and save the value in object_dict
            obj_key = b' '.join(parts[-2:]) + b' obj'
            object_dict[obj_key] = between_objects_data[space_pos - len(parts[-2]):end_pos]

    return object_dict


def find_substring_occurrences(substring, string):
    # Helper function to find all occurrences of a substring in a string
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            return
        yield start
        start += len(substring)  # use start += 1 to find overlapping matches


def get_pages(catalog):
    # Assuming catalog is a dictionary with a 'Pages' key
    return catalog.get('Pages', [])


def get_stream_data(xobject):
    # Assuming xobject is a dictionary with a 'Stream' key
    return xobject.get('Stream', b'')


def extract_base_elements(pdf_content, xref_position):
    # This is a simplified example and does not handle all cases or errors
    xref_table = parse_xref_table(pdf_content, xref_position)
    catalog = get_object(pdf_content, 1, xref_table)  # Assume object 1 is the catalog
    pages = get_pages(catalog)  # You'll need to write this function
    for page in pages:
        resources = page['Resources']
        fonts = resources['Font']
        for font in fonts.values():
            # Do something with font information
            pass
        xobjects = resources.get('XObject', {})
        for xobject in xobjects.values():
            # Check if the XObject is an image
            if xobject['Subtype'] == 'Image':
                image_data = get_stream_data(xobject)  # You'll need to write this function
                # Do something with image data
                pass
            elif xobject['Subtype'] == 'Form':
                # Recurse into form XObjects
                extract_base_elements(xobject['Resources'])


def parse_xref_table(pdf_content, xref_position):
    pos = xref_position
    xref_table = {}

    # Skip the 'xref' line and the object range line
    pos = pdf_content.find(b'\n', pos) + 1  # Skip 'xref'
    pos = pdf_content.find(b'\n', pos) + 1  # Skip object range

    while True:
        # Get the next line
        newline_pos = pdf_content.find(b'\n', pos)
        if newline_pos == -1:
            raise ValueError("Trailer not found")

        line = pdf_content[pos:newline_pos].decode('utf-8', 'ignore')

        if line.startswith('trailer'):
            # End of xref table
            break

        # Parse the fields based on their fixed widths
        obj_offset = int(line[:10])
        obj_gen_num = int(line[11:16])
        obj_status = line[17]

        # Assuming you want to store the offset of in-use objects
        if obj_status == 'n':
            obj_num = len(xref_table)  # Assuming objects are listed in order
            xref_table[obj_num] = obj_offset

        # Move to the next line
        pos = newline_pos + 1

    return xref_table


def handle_stream(lines):
    end_index = next((i for i, line in enumerate(lines) if line.endswith(b'endstream')), -1)
    if end_index != -1:
        stream_data = b'\n'.join(lines[:end_index])
        remaining_lines = lines[end_index + 1:]
        return stream_data, remaining_lines
    return None, lines


def parse_pdf_object(pdf_object):
    parsed_object = {
        'id': None,
        'props': {},
        'stream': None,
        'content': [],
        'type': None
    }

    lines = iter(pdf_object.split(b'\n'))

    for line in lines:
        if line.endswith(b'endobj'):
            break
        elif line.startswith(b'<<'):
            propsline = line
            props_lines = b""
            level = 0
            while True:
                try:
                    level += propsline.count(b'<<')  # Increment level by the count of '<<' markers
                    props_lines += propsline
                    level -= propsline.count(b'>>')  # Decrement level by the count of '>>' markers
                    if level == 0:
                        break
                    propsline = next(lines)
                except StopIteration:
                    break  # End of data reached before closing >>
            props = parse_properties(props_lines)
            if props:
                parsed_object['props'].update(props)
        elif line.endswith(b'stream'):
            stream_data, remaining_lines = handle_stream(list(lines))
            if stream_data:
                parsed_object['stream'] = stream_data
                lines = iter(remaining_lines)  # Update the lines iterator with the remaining lines
        elif b' obj' in line:
            parsed_object['id'] = line
        else:
            parsed_object['content'].append(line)

    return parsed_object


def parse_properties(prop_line):
    properties = {}
    i = 0
    stack = []
    current_key = b''
    current_value = b''
    while i < len(prop_line):
        if prop_line[i:i + 2] == b"<<":
            # Encountered the start of a nested dictionary.
            if current_key:
                # If there's a key, it means we're starting a nested dictionary.
                stack.append((properties, current_key))
                properties = {}  # Reset properties for the nested dictionary.
                current_key = b''  # Reset current_key.
            i += 1  # Skip the next character.
        elif prop_line[i:i + 2] == b">>":
            # Encountered the end of a nested dictionary.
            if stack:
                # If there's a stack, it means we're ending a nested dictionary.
                parent_properties, parent_key = stack.pop()
                parent_properties[parent_key] = properties
                properties = parent_properties  # Restore properties to the parent dictionary.
            i += 1  # Skip the next character.
        elif prop_line[i:i + 1] == b"/":
            # Encountered a new key or a new value.
            if current_key:
                # If there's a current_key, it means we've found a value.
                properties[current_key.decode()] = current_value.decode()
                current_key = b''  # Reset current_key.
                current_value = b''  # Reset current_value.
        elif prop_line[i:i + 1] == b" ":
            # Space separates keys from values.
            if not current_key:
                current_key = current_value  # If there's no current_key, current_value is actually the key.
                current_value = b''  # Reset current_value.
        else:
            if current_key:
                # Collect characters for keys or values.
                current_value += prop_line[i:i + 1]
            else:
                current_key += prop_line[i:i + 1]

        i += 1  # Move to the next character.

    # Handle any remaining key-value pair.
    if current_key:
        properties[current_key.decode()] = current_value.decode()
    elif current_value:
        # Handle a single key with no value.
        properties[current_value.decode()] = ''

    return properties


with open('../tests/data/Datasheet-Centaur-Charger-DE.6f.pdf', 'rb') as file:
    pdf_content = file.read()

header = parse_pdf_header(pdf_content)
xref_position = find_xref_position(pdf_content)
xref_table = parse_xref_table(pdf_content, xref_position)
object_dict = split_pdf_objects(pdf_content, xref_table)
object_props = {num: parse_pdf_object(value) for num, value in object_dict.items()}

import pandas as pd

df = pd.DataFrame(object_props).T.drop(columns="stream")

from parsimonious.grammar import Grammar, NodeVisitor

# example:
expr = '<</Type/Page/MediaBox [0 0 595 842]/Rotate 0/Parent 3 0 R/Resources<</ProcSet[/PDF /ImageC /Text]/ExtGState 18 0 R/XObject 19 0 R/Font 20 0 R>>/Contents 5 0 R>>'
pdf_object_grammer = Grammar(r"""
      Dictionary = "<<" Entry* ">>"
      Entry = Keyword Value
      Value = NestedDictionary / Array / Reference / Keyword / Number / String
      Keyword = "/" [A-Z 0-9]i+
      NestedDictionary = Dictionary
      String = ~"[A-Z 0-9]+"i
    """)

expr = '<</Type/Page/MediaBox [0 0 595 842]/Rotate 0/Parent 3 0 R/Resources<</ProcSet[/PDF /ImageC /Text]/ExtGState 18 0 R/XObject 19 0 R/Font 20 0 R>>/Contents 5 0 R>>'
pdf_object_grammer = Grammar(r"""
      Dictionary = "<<" Entry+ Dictionary* ">>"
      Entry = ~".+"i
    """)

pdf_object_grammer = Grammar(r"""
    Dictionary = "<<" (Dictionary / Entry)+ ">>"
    Entry = ~"[A-Z /]+"i
""")

pdf_object_grammer = Grammar(r"""
    Dictionary = "<<" Entry* ">>"
    Entry = Keyword Value
    Value = Keyword / Text / Dictionary
    Keyword = "/" Word
    Word = ~"[a-z0-9]+"i
    Text = ~"[ a-z0-9\[\]]+"i
""")

expr = '<</Type Page /MediaBox1 asdasdad1 /test<</MediaBox2 asdasd2 >>/MediaBox3 asdad3 >>'
expr = '<</Type/Page/MediaBox [0 0 595 842]/Rotate 0/Parent 3 0 R/Resources<</ProcSet[/PDF /ImageC /Text]/ExtGState 18 0 R/XObject 19 0 R/Font 20 0 R>>/Contents 5 0 R>>'
tree = pdf_object_grammer.parse(expr)
print(tree)


class Visitor(NodeVisitor):
    def generic_visit(self, node, visited_children):
        """ The generic visit method. """
        return visited_children or node


iv = Visitor()
output = iv.visit(tree)
print(output)
