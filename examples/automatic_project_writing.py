"""
Demonstrate how to load a directory into an index amd then write
about it.
"""

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
    # pydoxtools.DocumentBag uses a dask scheduler for parallel computing
    # in the background. For easier debugging, we set this to "synchronous"
    dask.config.set(scheduler='synchronous')
    # dask.config.set(scheduler='multiprocessing') # can als be used...

    settings.PDX_ENABLE_DISK_CACHE = True  # turn on caching for pydoxtools

    ##### Use chromadb as a vectorstore #####
    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(settings.PDX_CACHE_DIR_BASE / "chromadb"),
        anonymized_telemetry=False
    )

    # create our source of information. It creates a list of documents
    # in pydoxtools called "pydoxtools.DocumentBag" (which itself holds a list of pydoxtools.Document) and
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

    agent = ag.LLMAgent(
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
    agent.add_question(question="Can you please provide the main topic of the project or some primary "
                                "keywords related to the project, "
                                "to help with identifying the relevant files in the directory?",
                       answer="python library, AI, pipelines")

    # first, gather some basic information...
    questions = agent.execute_task(
        task="What additional information do you need to create a first, very short outline as a draft? " \
             "provide it as a ranked list of questions", save_task=True)
    # we only use he first 5 questions to make it faster ;).
    agent.research_questions(questions[:5], allowed_documents=["text/markdown"])

    # now write the text
    txt = agent.execute_task(task="Complete the overall objective, formulate the text "
                                  "based on answered questions and format it in markdown.",
                             context_size=20, max_tokens=1000, formatting="txt")
    final_result.append(txt)  # add a first draft to the result

    critique = agent.execute_task(task=f"Given this text:\n\n```markdown\n{txt}\n```"
                                       "\n\nlist 5 points of critique about the text",
                                  context_size=0, max_tokens=1000)

    tasks = agent.execute_task(
        task=f"Given this text:\n\n```markdown\n{txt}\n```\n\n"
             f"and its critique: {critique}\n\n"
             "Generate specific instructions for an AI to make the given text better. "
             "Sort them by importance and return them as a flat list of tasks",
        context_size=0, max_tokens=1000)

    for t in tasks:
        task = "Given this text:\n\n```markdown\n{txt}\n```\n\n" \
               f"Improve the text by modifying it according to this task: '{t}' " \
               f"Also pay attention to the overall objective. "
        txt = agent.execute_task(task, context_size=10, max_tokens=1000, formatting="markdown")
        final_result.append([task, txt])
