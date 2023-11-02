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


def parse_pdf_object(pdf_object):
    parsed_object = {
        'id': None,  # Add an 'id' field to hold the object identifier
        'props': {},
        'stream': None,
        'content': [],
        'type': None
    }

    lines = pdf_object.split(b'\n')
    inside_stream = False
    stream_data = []
    for line in lines:
        if line.endswith(b'endobj'):  # Check for the end of object marker
            break  # Exit the loop when the end of object marker is found
        elif line.endswith(b'endstream'):
            inside_stream = False
            parsed_object['stream'] = b''.join(stream_data)
            stream_data = []  # Reset stream data
        elif inside_stream:
            stream_data.append(line)
        elif line.endswith(b'stream'):
            inside_stream = True
        elif line.startswith(b'<<') and line.endswith(b'>>'):
            props = parse_properties(line)
            parsed_object['props'].update(props)
        elif b' obj' in line:
            parsed_object['id'] = line  # Capture the object identifier
        else:
            parsed_object['content'].append(line)

    return parsed_object


def parse_properties(prop_line):
    properties = {}
    tokens = prop_line.strip(b"<>/").split(b"/")

    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if b' ' in token:
            # If the token contains a space, it's a key-value pair.
            key, value = token.split(b' ', 1)
            properties[key] = value
        else:
            # Otherwise, it's a key, and the value is in the next token.
            key = token
            i += 1  # Move to the next token for the value.
            if i < len(tokens):  # Ensure we don't go out of bounds.
                value = tokens[i]
                properties[key] = value
        i += 1  # Move to the next token.

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
