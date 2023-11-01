def parse_pdf_header(pdf_content):
    # The PDF header is usually within the first 1024 bytes
    header_position = pdf_content.find(b'%PDF-')
    if header_position == -1:
        raise ValueError("Not a valid PDF file")
    header_end = pdf_content.find(b'\r\n', header_position)
    return pdf_content[header_position:header_end].decode()

def find_xref_position(pdf_content):
    # The xref position is typically near the end of the file
    xref_position = pdf_content.rfind(b'startxref')
    if xref_position == -1:
        raise ValueError("xref not found")
    xref_position_start = pdf_content.find(b'\r\n', xref_position) + 2
    xref_position_end = pdf_content.find(b'\r\n', xref_position_start)
    return int(pdf_content[xref_position_start:xref_position_end])

with open('../tests/data/Datasheet-Centaur-Charger-DE.6f.pdf', 'rb') as file:
    pdf_content = file.read()

header = parse_pdf_header(pdf_content)
xref_position = find_xref_position(pdf_content)

print(f"PDF Header: {header}")
print(f"xref Position: {xref_position}")
