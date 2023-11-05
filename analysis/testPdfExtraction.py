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


def extract_xrefs_and_trailers(pdf_content) -> list:
    # This function now returns a list of (xref, trailer) tuples
    xrefs_and_trailers = []
    current_pos = -1
    # Continue searching backwards for 'startxref' from the current position
    current_pos = pdf_content.rfind(b'startxref', 0, current_pos)
    if current_pos == -1:
        raise ValueError("no xref found")
    xref_start = int(pdf_content[current_pos + len(b'startxref\n'):].split()[0])
    while True:
        trailer_start = pdf_content.find(b'trailer', xref_start)
        trailer_end = pdf_content.find(b'>>', trailer_start) + 2
        if trailer_start == -1:
            raise ValueError("trailer start not found")
        xref = pdf_content[xref_start:trailer_start].strip()
        trailer = pdf_content[trailer_start:trailer_end].strip()
        xrefs_and_trailers.append((xref, trailer))

        # Find the previous xref offset directly after '/Prev' in the trailer
        prev_start = trailer.find(b'/Prev ') + 6
        if prev_start == 5:  # Not found (5 is from -1 + 6)
            break

        # Extract the integer value for the previous xref offset
        prev_end = prev_start
        while prev_end < len(trailer) and trailer[prev_end:prev_end + 1].isdigit():
            prev_end += 1
        prev_offset = int(trailer[prev_start:prev_end])

        # Set the current position to the previous xref offset for the next iteration
        xref_start = prev_offset

    # We reverse to maintain the order from oldest to newest
    return list(reversed(xrefs_and_trailers))


def split_pdf_objects(pdf_content, xref_table):
    object_dict = {}
    sorted_offsets = sorted(xref_table)

    for i, obj_offset in enumerate(sorted_offsets):
        obj_pos = obj_offset[0]
        if i + 1 < len(sorted_offsets):
            # If this is not the last known object, find the end position based on the next object's offset
            obj_end = sorted_offsets[i + 1][0]
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


def parse_xref(xref_bytes: bytes):
    table = []
    for line in xref_bytes.split(b"\n")[2:]:
        i1, i2, c = line.split()
        table.append((int(i1), int(i2), c))

    return table


def parse_xrefs(xrefs_and_trailers: list):
    object_dict = {}
    # Loop through all (xref, trailer) tuples
    for xref_bytes, trailer_bytes in xrefs_and_trailers:
        xref_table = parse_xref(xref_bytes)
        temp_object_dict = split_pdf_objects(pdf_content, xref_table)
        # Merge with the existing objects, newer entries will overwrite older ones
        object_dict.update(temp_object_dict)
    return object_dict


class Visitor(NodeVisitor):
    def visit_pdfobj(self, node, visited_children):
        obj_id, objcontent, _ = visited_children
        objcontent.update({"id": obj_id})
        return objcontent

    def visit_objcontent(self, node, visited_children):
        objs = {'props': {}}
        for c, *_ in visited_children:
            if c:
                if isinstance(c, dict):
                    objs['props'].update(c)
                else:
                    objs['value'] = c
        return objs

    def visit_props(self, node, visited_children):
        return visited_children[2]

    def visit_ws(self, node, visited_children):
        return None

    def visit_obj_id(self, node, visited_childred):
        int1, _, version, _, _ = visited_childred
        return (int1, version)

    def visit_int(self, node, visited_childred):
        return int(node.text)

    def visit_number(self, node, visited_children):
        return float(node.text)

    def visit_entry(self, node, visited_children):
        key, val = visited_children
        return (key, val)

    def visit_value(self, node, visited_children):
        return visited_children[2][0]

    def visit_string(self, node, visited_children):
        return visited_children[1]

    def visit_date(self, node, visited_children):
        return visited_children[1]

    def visit_reference(self, node, visited_children):
        int1, _, int2, _, _ = visited_children
        return (int1, int2, "R")

    def visit_list(self, node, visited_children):
        newlist = visited_children[1][0]
        return newlist

    def visit_array(self, node, visited_children):
        array = visited_children[1]
        # whitespace is usually "None" thats why we're checking...
        array = tuple(v[0][0] for v in array if (v[0] is not None))
        return array

    def visit_dict(self, node, visited_children):
        out = {k: v for k, v in visited_children}
        return out

    def visit_text(self, node, visited_children):
        return node.text.strip()

    def visit_unicodetext(self, node, visited_children):
        return node.text.strip()

    def visit_word(self, node, visited_childre):
        return node.text

    def visit_key(self, node, visited_children):
        word = visited_children[1].text
        return word

    def generic_visit(self, node, visited_children):
        """ The generic visit method. """
        return visited_children or node


pdf_obj_grammar = Grammar(r"""
    pdfobj = obj_id objcontent endobj
    objcontent = objentry*
    objentry = props / reference / list / int / ws
    props = "<<" ws? dict ws? ">>"
    dict = entry*
    entry = key value
    value = ws? "/"? (reference / int / text / list / props / word) ws?
    endobj = "endobj"
    list = ("[" / "(") (array / unicodetext) ("]" / ")")
    array = ws? ((reference / key / number / int / word) ws?){1,} ws?
    unicodetext = ~"[^\\)]*"
    key = "/" ~"[a-z0-9]+"is
    obj_id = int ws int ws "obj"
    reference = int ws int ws "R"
    text = (word / ws)+
    word = ~"[a-z0-9.\-:@\+']+"i
    int = ~"-?[0-9]+"
    number = ~"-?[0-9]+\.[0-9]+"
    ws = ~"\s+"
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

    try:
        # Assuming the remaining content is well-formed and can be parsed by your grammar and visitor
        tree = pdf_obj_grammar.parse(remaining_content.decode('utf-8', 'ignore'))
        output = iv.visit(tree)
    except Exception as e:
        raise TypeError(f'we got a problem with this string:\n\n{remaining_content}\n\n')

    # Assuming the output is a dictionary, merge it with the parsed_object dictionary
    parsed_object.update(output)

    return parsed_object


# def test_files():
if True:
    files = [
        '../tests/data/Datasheet-Centaur-Charger-DE.6f.pdf',
        '../tests/data/PFR-PR23_BAT-110__V1.00_.pdf'
    ]

    for f in files:
        with open(f, 'rb') as file:
            pdf_content = file.read()

        header = parse_pdf_header(pdf_content)
        xref = extract_xrefs_and_trailers(pdf_content)
        object_dict = parse_xrefs(xref)

        # df = pd.DataFrame(object_props).T.drop(columns="stream")

        # def test_object_extraction():
        texts = (
            b"31 0 obj\r\n<</Author(\xfe\xff\x00B\x00j\x00\xf6\x00r\x00n\x00 \x00D\x00a\x00n\x00z\x00i\x00g\x00e\x00r) /Creator(\xfe\xff\x00M\x00i\x00c\x00r\x00o\x00s\x00o\x00f\x00t\x00\xae\x00 \x00W\x00o\x00r\x00d\x00 \x00f\x00\xfc\x00r\x00 \x00O\x00f\x00f\x00i\x00c\x00e\x00 \x003\x006\x005) /CreationDate(D:20200722152437+02'00') /ModDate(D:20200722152437+02'00') /Producer(\xfe\xff\x00M\x00i\x00c\x00r\x00o\x00s\x00o\x00f\x00t\x00\xae\x00 \x00W\x00o\x00r\x00d\x00 \x00f\x00\xfc\x00r\x00 \x00O\x00f\x00f\x00i\x00c\x00e\x00 \x003\x006\x005) >>\r\nendobj"
            ,
            b'18 0 obj\r\n[ 19 0 R] \r\nendobj'
            ,
            b'3 0 obj\n<< /Type /Pages /Kids [\n4 0 R\n21 0 R\n] /Count 2\n/Rotate 0>>\nendobj'
            ,
            b'11 0 obj\n<</BaseFont/XRQUZW+MyriadPro-Regular/FontDescriptor 10 0 R/Type/Font\n/FirstChar 1/LastChar 86/Widths[ 212 736 207 481 234\n448 331 327 549 555 ]\n/Encoding 44 0 R/Subtype/Type1>>\nendobj'
            ,
            b'11 0 obj\n<</BaseFont/XRQUZW+MyriadPro-Regular/FontDescriptor 10 0 R/Type/Font\n/FirstChar 1/LastChar 86/Widths[ 212 736 207 481 234\n448 331 327 549 555 ]\n/Encoding 44 0 R/Subtype/Type1>>\nendobj'
            ,
            b'17 0 obj\r\n<</Type/Font/Subtype/Type0/BaseFont/ArialMT/Encoding/Identity-H/DescendantFonts 18 0 R/ToUnicode 209 0 R>>\r\nendobj'
            ,
            b'13 0 obj\r\n<</Subtype/Link/Rect[ 307.54 52.669 404.42 67.927] /BS<</W 0>>/F 4/A<</Type/Action/S/URI/URI(mailto:info@berlin-space-tech.com) >>>>\r\nendobj'
            ,
            b'3 0 obj\r\n<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 5 0 R/F2 9 0 R/F3 11 0 R/F4 15 0 R/F5 17 0 R/F6 22 0 R>>/ExtGState<</GS7 7 0 R/GS8 8 0 R>>/XObject<</Image14 14 0 R/Image24 24 0 R>>/ProcSet[/PDF/Text/ImageB/ImageC/ImageI] >>/Annots[ 13 0 R] /MediaBox[ 0 0 595.32 841.92] /Contents 4 0 R/Group<</Type/Group/S/Transparency/CS/DeviceRGB>>/Tabs/S/StructParents 0>>\r\nendobj'
            ,
            b'11 0 obj\n<</BaseFont/XRQUZW+MyriadPro-Regular/FontDescriptor 10 0 R/Type/Font\n/FirstChar 1/LastChar 86/Widths[ 212 736 207 481 234 448 331 327 549 555 501 559 471 834 558\n492 542 239 666 532 482 236 513 513 513 370 646 612 497 555 658\n564 396 569 207 596 284 513 284 513 513 513 487 463 513 307 737\n551 482 569 207 292 428 493 652 469 472 542 551 548 804 513 580\n792 343 318 243 553 549 239 207 207 846 354 354 513 500 563 647\n647 538 612 605 311 689 501]\n/Encoding 44 0 R/Subtype/Type1>>\nendobj'
            ,
            b'25 0 obj\n<</BaseFont/SNDABN+Helvetica/FontDescriptor 24 0 R/Type/Font\n/FirstChar 32/LastChar 32/Widths[\n278]\n/Encoding/WinAnsiEncoding/Subtype/Type1>>\nendobj'
            ,
            b'24 0 obj\n<</Type/FontDescriptor/FontName/SNDABN+Helvetica/FontBBox[0 0 1000 1000]/Flags 5\n/Ascent 0\n/CapHeight 0\n/Descent 0\n/ItalicAngle 0\n/StemV 0\n/AvgWidth 278\n/MaxWidth 278\n/MissingWidth 278\n/CharSet(/space)/FontFile3 34 0 R>>\nendobj'
            ,
            b'10 0 obj\n<</Type/FontDescriptor/FontName/XRQUZW+MyriadPro-Regular/FontBBox[-46 -250 838 837]/Flags 4\n/Ascent 837\n/CapHeight 837\n/Descent -250\n/ItalicAngle 0\n/StemV 125\n/MissingWidth 500\n/CharSet(/two/L/quoteright/A/germandbls/y/quotedblleft/n/c/three/M/parenleft/B/z/o/d/four/N/parenright/C/p/e/at/Z/five/O/D/udieresis/quotedblright/adieresis/q/f/six/P/plus/E/space/r/g/seven/comma/F/s/h/eight/degree/hyphen/R/G/egrave/endash/t/i/nine/S/period/H/u/j/Adieresis/colon/T/slash/I/Udieresis/v/k/quoteleft/U/zero/J/percent/odieresis/twosuperior/w/l/a/V/one/K/ampersand/x/m/bar/b/W)/FontFile3 36 0 R>>\nendobj'
            ,
            b'2 0 obj\n<</Producer(GPL Ghostscript 8.15)\n/CreationDate(D:20140311091801)\n/ModDate(D:20140311091801)\n/Title(Microsoft Word - Datasheet - Centaur Charger - rev 05 - DE.docx)\n/Creator(PScript5.dll Version 5.2.2)\n/Author(marianka pranger)>>endobj'
        )
        for txt in texts[:]:
            txtd = txt.decode('utf-8', 'ignore')
            tree = pdf_obj_grammar.parse(txtd)
            out = iv.visit(tree)

        # test_object_extraction()

        object_props = {num: parse_pdf_object(value) for num, value in object_dict.items()}
