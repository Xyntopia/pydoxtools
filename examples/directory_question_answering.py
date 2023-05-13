"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import logging

import chromadb
import dask
import pandas as pd
from chromadb.config import Settings
from dask.diagnostics import ProgressBar

from pydoxtools import DocumentBag
from pydoxtools.extract_nlpchat import openai_chat_completion
from pydoxtools.settings import settings

if __name__ == "__main__":
    # from dask.distributed import Client
    # from dask.distributed import Client
    # client = Client()
    # dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    # print(client.scheduler_info())

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

    # settings.PDX_ENABLE_DISK_CACHE = True
    # dask.config.set(scheduler='multiprocessing')  # overwrite default with single-threaded scheduler for debugging

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(settings.PDX_CACHE_DIR_BASE / "chromadb")
    ))

    root_dir = "../../pydoxtools"
    ds = DocumentBag(
        source=root_dir,
        exclude=[
            '.git/', '.idea/', '/node_modules', '/dist',
            '/__pycache__/', '.pytest_cache/', '.chroma', '.svg', '.lock',
            "/site/"
        ])  # .config(verbosity="v")

    # ds = DocumentBag(['../.git', '../README.md', '../DEVELOPMENT.md', '../docker-compose.yml'])
    # ds.take(10)[2].file_meta

    # idx = ds.e('text_segments')
    idx = ds.e('text_segments', meta_properties=["file_meta"])

    collection = client.get_or_create_collection(name="index")
    compute, query = idx.add_to_chroma(collection)  # remove number to calculate for all files!
    need_index = True
    if need_index:
        # collection = client.get_collection(name="index")
        client.delete_collection(name="index")
        collection = client.get_or_create_collection(name="index")
        with ProgressBar():
            # idx.idx_dict.take(200, npartitions=3)  # remove number to calculate for all files!
            # idx.idx_dict.compute()
            compute()

    client.persist()

    ######  experiment with the writing process  #####

    # TODO: missing information:  ProjectType, generate a question to find out the project type from user input.
    # TODO: also rank directories

    stored_infos = []

    question: "What type of a project is this?"
    question: "Can you give some keywords for this project?"

    stored_infos.append({"project type": "python library"})
    stored_infos.append({"project keywords": "python library, AI, pipelines"})

    objective = "Write a blog post, about 1 page long, introducing this new library"

    query("readme")

    df = pd.DataFrame([f.suffix for f in ds.file_path_list])
    df.value_counts()
    files = "\n".join([f for f in df[0].unique()])

    task = f"rank the file-endings which are most relevant for finding " \
           f"a project description, readme, introduction, a manual or similar text:\n" \
           f"\nlist the top 3 in descending order:\n1: X1 2: X2 3: ... "

    input_data = f"{files}"


    def task_msgs(objective, stored_infos, task, input_data):
        return ({"role": "system",
                 "content": "You are a helpful assistant that aims to complete the given task."},
                {"role": "user", "content": f"# Overall objective: \n{objective}\n\n"
                                            f"# Context: \n{stored_infos} \n\n"
                                            f"# Task: \n{task} \n\n"
                                            f"# Input for the current task: \n{input_data}.\n\n"
                                            f"# Now write the answer, without any additional text, under any circumstances:"})


    msgs = task_msgs(objective, stored_infos, task, input_data)
    res = openai_chat_completion(msgs, max_tokens=256).content
    important_files = res.split()
    stored_infos.append({"important file endings for documentation": res})

    query("open source software project name in machine learning, artificial intelligence or data analysis")
