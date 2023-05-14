"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import logging
import uuid

import chromadb
import dask
import pandas as pd
import yaml
from chromadb.config import Settings
from dask.diagnostics import ProgressBar

from pydoxtools import DocumentBag, Document
from pydoxtools.extract_nlpchat import openai_chat_completion
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)


def generate_function_usage_prompt(func: callable):
    # TODO: summarize function usage, only extract the parameter description
    # func.__doc__
    return func.__doc__


def add_info_to_collection(collection, doc: Document, type: str):
    collection.add(
        # TODO: embeddings=  #use our own embeddings for specific purposes...
        embeddings=[doc.embedding],
        documents=[doc.full_text],
        metadatas=[{"information_type": type}],
        ids=[uuid.uuid4().hex]
    )


def task_chat(objective, context, task):
    return ({"role": "system",
             "content": "You are a helpful assistant that aims to complete one task."},
            {"role": "user", "content": f"# Overall objective: \n{objective}\n\n"
                                        f"# Take into account these previously completed tasks: "
                                        f"\n{context} \n\n"
                                        f"# This is the task: \n{task} \n\n"
                                        f"# Return the answer without any additional text, "
                                        f"under any circumstances, in yaml format."
                                        f" RESPONSE: "})


def get_context(task: str):
    context = query(task, where={
        "$or": [{"information_type": "question"},
                {"information_type": "task"}]})["documents"]
    return context


def execute_task(objective, task):
    context = query(task, where={
        "$or": [{"information_type": "question"},
                {"information_type": "task"}]})["documents"]
    msgs = task_chat(objective, context,
                     task=task)
    logger.info(f"execute_task: {msgs[1]['content']}")
    res = openai_chat_completion(msgs, max_tokens=256).content
    task_pair = Document({"task": task, "result": res})
    add_info_to_collection(collection, task_pair, "task")
    return res


# TODO: index all of the functions here to make sure, we can
#       use them with chatgpt
Document.pipeline_docs()

if __name__ == "__main__":
    # from dask.distributed import Client
    # from dask.distributed import Client
    # client = Client()
    # dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler for debugging
    # print(client.scheduler_info())

    settings.PDX_ENABLE_DISK_CACHE = True
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
    need_index = False
    if need_index:
        # collection = client.get_collection(name="index")
        client.delete_collection(name="index")
        collection = client.get_or_create_collection(name="index")
        with ProgressBar():
            # idx.idx_dict.take(200, npartitions=3)  # remove number to calculate for all files!
            # idx.idx_dict.compute()
            compute()

    client.persist()

    ## this is only for debugging:
    collection.delete(
        where={"information_type": "question"}
    )
    collection.delete(
        where={"information_type": "task"}
    )
    stored_info = \
        collection.get(
            where={"$or": [{"information_type": "question"},
                           {"information_type": "task"}]})["documents"]

    ######  experiment with the writing process  #####

    objective = "Write a blog post, about 1 page long, introducing this new library"

    question = "Can you please provide the main topic of the project or some primary " \
               "keywords related to the project, " \
               "to help with identifying the relevant files in the directory?"
    answer = "python library, AI, pipelines"
    qa_pair = {"question ": question.strip(),
               "answer": answer.strip()}
    doc = Document(qa_pair)
    add_info_to_collection(collection, doc, type="question")

    task = "What additional information do you need to create a first outline as a draft? " \
           "provide it as a list"
    res = execute_task(objective=objective, task=task)
    important_things = yaml.unsafe_load(res)

    raise

    topic = important_things[0]
    task = "Create a list of input strings that we can use for an embeddings " \
           f"based search to find text snippets to get information about the topic: {topic}"
    res = execute_task(objective=objective, task=task)
    list_of_search_strings = yaml.unsafe_load(res)

    df = pd.DataFrame([f.suffix for f in ds.file_path_list])
    df.value_counts()
    file_endings = "\n".join([f for f in df[0].unique()])

    task = f"rank the file-endings which are most relevant for finding " \
           f"a project description, readme, introduction, a manual or similar text. " \
           f"List the top 5 in descending order, file_endings: {file_endings}"
    res = execute_task(objective, task)
    ranked_file_endings = yaml.unsafe_load(res)

    ds.d("filename").compute()
    filtered_files = DocumentBag(ds.paths(
        max_depth=2,
        mode="files",
        wildcard="*" + ranked_file_endings[0]
    )).d("path").compute()

    task = "Create a first outline for the provided objective"
    res = execute_task(objective=objective, task=task)
    important_things = yaml.unsafe_load(res)

    task = "Create a list of 10 most important things that we need to find information" \
           " for, so that we can fulfill the objective"
    res = execute_task(objective=objective, task=task)
    important_things = yaml.unsafe_load(res)

    task = "Create a list of input strings that we can use for an embeddings " \
           "based search to find text snippets to get information"
    res = execute_task(objective=objective, task=task)
    list_of_search_strings = yaml.unsafe_load(res)

    get_context(task)

    q = list_of_search_strings[4]
    for q in list_of_search_strings:
        query(q)

    # task = "Given the list of files, create a ranking of what we should look at first"

    # TODO: create a task directly based on this:
    #           collection.get.__doc__
    # task = f""
    # get context
    # collection.get(
    #    query
    #    Document
    # )

    # collection.get(
    #    query
    #    Document
    # )

    # TODO: run our task loop with critique...  trying to fulfill the tasks...
    # max_steps = 10
    # for i in range(max_steps):
    #    msgs = task_msgs(objective, stored_infos, task, input_data)
    #    res = openai_chat_completion(msgs, max_tokens=256).content
