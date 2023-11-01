import re


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


def extract_objects(pdf_content, xref_table):
    objects = []
    for obj_num, obj_offset in xref_table.items():
        obj_dict = {}
        obj_pos = obj_offset
        # Find the end of the object (assumes the next object starts with its object number)
        obj_end = pdf_content.find(f'{obj_num + 1} 0 obj'.encode(), obj_pos)
        if obj_end == -1:
            # If this is the last object, find the 'endobj' keyword to determine the end of the object
            obj_end = pdf_content.find(b'endobj', obj_pos + 1) + len('endobj')
        obj_data = pdf_content[obj_pos:obj_end].decode('utf-8', 'ignore')
        # Split the object data into lines and process each line
        for line in obj_data.split('\n'):
            if line.startswith('<<'):
                # This is the start of a dictionary; parse the dictionary
                dict_end = obj_data.find('>>', obj_pos) + len('>>')
                dict_data = obj_data[obj_pos:dict_end]
                # Assume each line contains one key-value pair
                for dict_line in dict_data.split('\n'):
                    key_value = dict_line.split(' ', 1)
                    if len(key_value) == 2:
                        obj_dict[key_value[0]] = key_value[1]
                obj_pos = dict_end  # Update the position to the end of the dictionary
            elif line.startswith('stream'):
                # This is the start of a stream; find the end of the stream
                stream_end = obj_data.find('endstream', obj_pos) + len('endstream')
                stream_data = obj_data[obj_pos:stream_end]
                obj_dict['Stream'] = stream_data
                obj_pos = stream_end  # Update the position to the end of the stream
        objects.append(obj_dict)
    return objects


def split_pdf_objects(pdf_content, xref_table):
    object_dict = {}
    sorted_offsets = sorted(list(xref_table.values()))

    for i, obj_offset in enumerate(sorted_offsets):
        obj_pos = obj_offset
        if i + 1 < len(sorted_offsets):
            # If this is not the last known object, find the end position based on the next object's offset
            obj_end = sorted_offsets[i + 1]
        else:
            # TODO: this is probably not the end, as there can be multiple objects! we would
            #       have to look for the "last" endobj  I assume...
            # If this is the last known object, find the 'endobj' keyword to determine the end of the object
            obj_end = pdf_content.find(b'endobj', obj_pos) + len('endobj')
            if obj_end == -1 + len('endobj'):  # if 'endobj' is not found
                obj_end = len(pdf_content)  # use the end of the pdf content

        # Extract objects between current object and next object or end of file
        between_objects_data = pdf_content[obj_pos:obj_end]
        for match in re.finditer(rb'(\d+ \d+ obj)', between_objects_data):
            start_pos = match.start()
            end_pos = between_objects_data.find(b'endobj', start_pos) + len('endobj')
            if end_pos != -1 + len('endobj'):  # if 'endobj' is found
                obj_num = int(match.group(1).split()[0])
                object_dict[obj_num] = between_objects_data[start_pos:end_pos]

    return object_dict


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


with open('../tests/data/Datasheet-Centaur-Charger-DE.6f.pdf', 'rb') as file:
    pdf_content = file.read()

header = parse_pdf_header(pdf_content)
xref_position = find_xref_position(pdf_content)

print(f"PDF Header: {header}\n\n")
print(f"xref Position: {xref_position}\n\n")
print(pdf_content[xref_position:xref_position + 100].decode('utf-8', 'ignore'))
print(pdf_content[xref_position:].decode('utf-8', 'ignore'))

xref_table = parse_xref_table(pdf_content, xref_position)

# Assuming xref_table has been populated and pdf_content contains the PDF data
objects_list = extract_objects(pdf_content, xref_table)

# Usage:
# Assuming xref_table has been populated and pdf_content contains the PDF data
object_dict = split_pdf_objects(pdf_content, xref_table)
