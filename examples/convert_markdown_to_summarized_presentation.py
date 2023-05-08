"""
Short example on how to convert a wikipedia article into a presentation!

It only shortens the text for each section and then makes a slide for each section...
"""

import pydoxtools as pdx

# TODO: rewrite this as a oneliner, using DocumentBag and vectorizers and maybe chatgpt...

d = pdx.Document("../../tests/data/Starship.md")
df = d.text_box_elements
short_text = {}
for title, text in d.sections.items():
    print(f"summarizing chapter: {title}")
    sec_sum = pdx.Document(f"""{title}
    
    {text}
    """).slow_summary
    short_text[title] = sec_sum

text = "\n\n".join(
    f"""## {title}
    
{text}""" for title, text in short_text.items())

with open("Starship.pptx", "wb") as f:
    pptx_str = pdx.Document(text, document_type="text/markdown").convert_to("pptx")
    f.write(pptx_str)
