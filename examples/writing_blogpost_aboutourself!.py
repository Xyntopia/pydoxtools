"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import logging

import dask
from chromadb.config import Settings

import pydoxtools as pdx
from pydoxtools import agent as ag
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydoxtools.document").setLevel(logging.INFO)

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

    # create our source of information. It creates a list of documents
    # in pydoxtools called "DocumentBag" and
    # here we choose to use pydoxtools itself as an information source!
    root_dir = "../../pydoxtools"
    ds = pdx.DocumentBag(
        source=root_dir,
        exclude=[  # ignore some files which make the indexing rather inefficient
            '.git/', '.idea/', '/node_modules', '/dist',
            '/__pycache__/', '.pytest_cache/', '.chroma', '.svg', '.lock',
            "/site/"
        ],
        forgiving_extracts=True
    )

    ######  Start the writing process  #####
    final_result = []

    agent = ag.AgentBase(
        vector_store=chroma_settings,
        objective="Write a blog post, introducing a new library (which was developed by us, "
                  "the company 'Xyntopia') to "
                  "visitors of our corporate webpage, which might want to use the pydoxtools library but "
                  "have no idea about programming. Make sure, the text is about half a page long.",
        data_source=ds
    )
    agent.pre_compute_index()

    # first, add a basic answer, to get the algorithm started a bit more quickly :) we could gather this information
    # from the CLI for example...
    agent.add_question(question="Can you please provide the main topic of the project or some primary " \
                                "keywords related to the project, " \
                                "to help with identifying the relevant files in the directory?",
                       answer="python library, AI, pipelines")

    task = "What additional information do you need to create a first, very short outline as a draft? " \
           "provide it as a ranked list of questions"
    res = agent.execute_task(task)
    questions = ag.yaml_loader(res)[:5]
    agent.add_task(task, str(questions))
    agent.research_questions(questions)

    # now write the text
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
