import pydoxtools as pdx

if __name__ == "__main__":
    # create a document from a file, string, bytestring, file-like object
    # or even an url:
    doc = pdx.Document(
        "https://www.raspberrypi.org/app/uploads/2012/12/quick-start-guide-v1.1.pdf",
        document_type=".pdf"
    )
    # extract the table as a pandas dataframe:
    print(doc.tables_df)
    # ask a question about the document, using Q&A Models (questionas answered locally!):
    print(doc.answers(["how much power does it need?"])[0][0][0])
    # ask a question about the document, using ChatGPT:
    print(doc.chat_answers(["who is the target group of this document?"])[0].content)
    print(doc.chat_answers(["Answer if a 5-year old would be able to follow these instructions?"])[0].content)
