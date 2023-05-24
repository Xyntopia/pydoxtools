"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import logging

import chromadb
import dask
import numpy as np
from chromadb.config import Settings
from dask.diagnostics import ProgressBar

from pydoxtools import DocumentBag, Document
from pydoxtools import agent as ag
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)


def stophere():
    raise NotImplementedError()


"""
TODO: integrate this...
df = pd.DataFrame([f.suffix for f in ds.file_path_list])
df.value_counts()
file_endings = "\n".join([f for f in df[0].unique()])

task = f"rank the file-endings which are most relevant for finding " \
       f"a project description, readme, introduction, a manual or similar text. " \
       f"List the top 5 in descending order, file_endings: {file_endings}"
res = execute_task(objective, task)
ranked_file_endings = yaml_loader(res)

ds.d("filename").compute()
filtered_files = DocumentBag(ds.paths(
    max_depth=2,
    mode="files",
    wildcard="*" + ranked_file_endings[0]
)).d("path").compute()
"""

if __name__ == "__main__":
    # from dask.distributed import Client
    # from dask.distributed import Client
    # client = Client()
    dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    # dask.config.set(scheduler='threading')  # overwrite default with single-threaded scheduler for debugging
    # print(client.scheduler_info())

    settings.PDX_ENABLE_DISK_CACHE = True
    # dask.config.set(scheduler='multiprocessing')  # overwrite default with single-threaded scheduler for debugging

    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(settings.PDX_CACHE_DIR_BASE / "chromadb"),
        anonymized_telemetry=False
    )
    collection_name = "blog_index"
    client = chromadb.Client(chroma_settings)
    info_collection = client.get_or_create_collection(name=collection_name)

    root_dir = "../../pydoxtools"
    ds = DocumentBag(
        source=root_dir,
        exclude=[
            '.git/', '.idea/', '/node_modules', '/dist',
            '/__pycache__/', '.pytest_cache/', '.chroma', '.svg', '.lock',
            "/site/"
        ],
        forgiving_extracts=True
    )

    idx = ds.apply(new_document='text_segments', document_metas="file_meta")
    idx = idx.exploded
    compute, _ = idx.add_to_chroma(
        chroma_settings, collection_name)  # remove number to calculate for all files!

    # create the index if we run it for the first time!
    # need_index = False if collection.count() > 0 else True
    need_index = False
    if need_index:
        # collection = client.get_collection(name="index")
        client.delete_collection(name=collection_name)
        client.persist()
        collection = client.get_or_create_collection(name=collection_name)
        with ProgressBar():
            # idx.idx_dict.take(200, npartitions=3)  # remove number to calculate for all files!
            # idx.idx_dict.compute()
            compute()

    # after saving everything we need to re-load the client/collection back into memory...
    client = chromadb.Client(chroma_settings)
    info_collection = client.get_or_create_collection(name=collection_name)

    ######  Start the writing process  #####
    final_result = []

    info_collection.delete(where=ag.AgentBase._context_where_all)
    # TODO: do this in a cli..
    agent = ag.AgentBase(
        chromadb_collection=info_collection,
        objective="Write a blog post, introducing a new library (which was developed by us, "
                  "the company 'Xyntopia') to " \
                  "visitors of our corporate webpage, which might want to use the pydoxtools library but "
                  "have no idea about programming. Make sure, the text is about half a page long.",
        documents=ds
    )

    info_collection.upsert(
        embeddings=[Document("alright, what is this").embedding.tolist()],
        documents=["test"],
        metadatas=[{"test": "True"}],
        ids=["12345"]
    )

    # first, add a basic answer, to get the algorithm startet more quickly :).
    agent.add_question(question="Can you please provide the main topic of the project or some primary " \
                                "keywords related to the project, " \
                                "to help with identifying the relevant files in the directory?",
                       answer="python library, AI, pipelines")

    task = "What additional information do you need to create a first, very short outline as a draft? " \
           "provide it as a ranked list of questions"
    res = agent.execute_task(task)
    questions = ag.yaml_loader(res)[:5]
    agent.add_task(task, str(questions))

    # research question in index...
    for question in questions:
        agent.research_question(question)

    long_text = False  # longer than certain max_tokens
    if long_text:
        task = "Create an outline for our objective and format it as a flat yaml dictionary," \
               "indicating the desired number of words for each section."
        res = agent.execute_task(task)
        outline = ag.yaml_loader(res)
        agent.add_task(task, res)

        sections = []
        for section_name, word_num in outline.items():
            task = f"Create a list of keywords for the section: " \
                   f"{section_name} with approx {word_num} words"
            res = agent.execute_task(task)
            section_keywords = ag.yaml_loader(res)

            # take the mean of all the above search strings :)
            embedding = np.mean([Document(q).embedding for q in section_keywords], 0).tolist()
            # make sure we have no duplicate results
            res = query(embeddings=embedding, where={"num_words": {"$gt": 10}})
            txt = "\n\n".join(list(set(res["documents"][0])))

            task = f"Generate text for the section '{section_name}' with the keywords {section_keywords}. " \
                   f"and this size: {word_num}. " \
                   f"The following text sections can give some context. Use them if it makes sense" \
                   f" to do so:```\n\n{txt}```" \
                   f" the result should have a heading and be formated in markdown"
            res = agent.execute_task(task, context_size=0)
            section_text = ag.yaml_loader(res)

    task = "Complete the overall objective, formulate the text based on answered questions" \
           " and format it in markdown."
    res = agent.execute_task(task, context_size=20, max_tokens=1000)
    txt = res
    final_result.append(txt)

    task = "Given this text:\n\n" \
           f"```markdown\n{txt}\n```" \
           "\n\nlist 5 points of " \
           "critique about the text"
    res = agent.execute_task(task, context_size=0, max_tokens=1000)
    critique = ag.yaml_loader(res)

    task = "Given this text:\n\n" \
           f"```markdown\n{txt}\n```\n\n" \
           f"and its critique: {critique}\n\n" \
           "Generate instructions " \
           "that would make it better. Sort them by importance and return" \
           " it as a list of tasks"
    res = agent.execute_task(task, context_size=0, max_tokens=1000)
    tasks = ag.yaml_loader(res)

    for t in tasks:
        task = "Given this text:\n\n" \
               f"```markdown\n{txt}\n```\n\n" \
               f"Make the text better by executing this task: '{t}' " \
               f"and integrate it into the given text, but keep the overall objective in mind."
        txt = agent.execute_task(task, context_size=10, max_tokens=1000, format="markdown")
        final_result.append(res)
        # tasks = yaml_loader(res)

    # test the final result
    task = "Given this text:\n\n" \
           f"```markdown\n{txt}\n```\n\n" \
           f"Is it too long?"
    res = agent.execute_task(task, context_size=0, max_tokens=1000, format="markdown")
