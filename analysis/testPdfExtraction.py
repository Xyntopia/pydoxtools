from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


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


class Visitor(NodeVisitor):
    def visit_pdfobj(self, node, visited_children):
        obj_id, objcontent, _ = visited_children
        objcontent.update({"id": obj_id})
        return objcontent

    def visit_objcontent(self, node, visited_children):
        objs = {}
        for c,*_ in visited_children:
            if c:
                if isinstance(c, int):
                    objs['value']=c
                else:
                    objs[c[0]]=c[1]
        return objs

    def visit_ws(self, node, visited_children):
        return None

    def visit_obj_id(self, node, visited_childred):
        int1, _, version, _, _ = visited_childred
        return (int1, version)

    def visit_int(self, node, visited_childred):
        return int(node.text)

    def visit_props(self, node, visited_children):
        return ("props", visited_children[1])

    def visit_entry(self, node, visited_children):
        key, value = visited_children[0], visited_children[-1][0]
        return (key, value)

    def visit_list(self, node, visited_children):
        newlist = tuple(t[0] for t in visited_children[1] if isinstance(t[0], str))
        return newlist

    def visit_dict(self, node, visited_children):
        out = {k: v for k, v in visited_children}
        return out

    def visit_text(self, node, visited_children):
        return node.text

    def visit_word(self, node, visited_childre):
        return node.text

    def visit_key(self, node, visited_children):
        _, word = visited_children
        return word

    def generic_visit(self, node, visited_children):
        """ The generic visit method. """
        return visited_children or node


pdf_obj_grammar = Grammar(r"""
    pdfobj = obj_id objcontent "endobj"
    objcontent = objentry*
    objentry = props / int / ws
    obj_id = int ws int ws "obj"
    props = "<<" dict ">>"
    dict = entry*
    entry = key ws? value
    value = list / key / text / props
    list = lpar array rpar
    array = (key / word / ws)+
    key = "/" word
    word = ~"[a-z0-9]+"i
    text = ~"[ a-z0-9]+"i
    int = ~"[0-9]+"
    ws = ~"\s*"
    lpar  = "["
    rpar  = "]"
""")

iv = Visitor()


def extract_stream_data(content: bytes) -> [bytes, bytes]:
    stream_start = content.find(b'stream')
    if stream_start == -1:
        return None, content  # No stream found
    stream_start += len(b'stream\n')  # Skip the 'stream\n' keyword
    stream_end = content.find(b'endstream', stream_start)
    if stream_end == -1:
        return None, content  # No endstream found
    stream_data = content[stream_start:stream_end]
    remaining_content = content[:stream_start - len(b'stream\n')] + content[stream_end + len(b'endstream'):]
    return stream_data, remaining_content


def parse_pdf_object(pdf_object):
    parsed_object = {}

    # Handle the stream part
    stream_data, remaining_content = extract_stream_data(pdf_object)
    parsed_object['stream'] = stream_data

    # Assuming the remaining content is well-formed and can be parsed by your grammar and visitor
    tree = pdf_obj_grammar.parse(remaining_content.decode('utf-8', 'ignore'))
    output = iv.visit(tree)

    # Assuming the output is a dictionary, merge it with the parsed_object dictionary
    parsed_object.update(output)

    return parsed_object


with open('../tests/data/Datasheet-Centaur-Charger-DE.6f.pdf', 'rb') as file:
    pdf_content = file.read()

header = parse_pdf_header(pdf_content)
xref_position = find_xref_position(pdf_content)
xref_table = parse_xref_table(pdf_content, xref_position)
object_dict = split_pdf_objects(pdf_content, xref_table)
object_props = {num: parse_pdf_object(value) for num, value in object_dict.items()}
# df = pd.DataFrame(object_props).T.drop(columns="stream")


txt = '6 0 obj\n24535\nendobj'
txt = object_dict[b'5 0 obj'][:100] + object_dict[b'5 0 obj'][-100:]
txt = txt.decode('utf-8', 'ignore')
txt = r"""5 0 obj
<</Length 6 0 R/Filter /FlateDecode>>
stream
x/q\x1ev(JKqCP/EP?%XX\x17k\x1fN\x04iL֢ \x1ciL(,㒇պ$w\x0f;|gL^ooI\x05zh
v\x0c+y\x13*y@A\x08:k|\x00hPcendstream
endobj"""
txt = """5 0 obj
<</Length 6 0 R/Filter /FlateDecode>>
stream
asdasdasxdasdasdaendstream
endobj"""
txt = b'5 0 obj\n<</Length 6 0 R/Filter /FlateDecode>>\n\nendobj'
tree = pdf_obj_grammar.parse(txt)
print(tree)

out = iv.visit(tree)

print(out[(5, 0)])
