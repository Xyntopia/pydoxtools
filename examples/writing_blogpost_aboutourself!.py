"""Demonstrate how to load a directory into an index and
then answer questions about it"""

import logging
import uuid

import chromadb
import dask
import numpy as np
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


def stophere():
    raise NotImplementedError()


def generate_function_usage_prompt(func: callable):
    # TODO: summarize function usage, only extract the parameter description
    # func.__doc__
    return func.__doc__


def yaml_loader(txt: str):
    # Remove ```yaml and ``` if they are present in the string
    txt = txt.strip().replace("```yaml", "").replace("```", "")
    txt = txt.strip("`")
    data = yaml.unsafe_load(txt)
    return data


def add_info_to_collection(collection, doc: Document, metas: list[dict]):
    collection.add(
        # TODO: embeddings=  #use our own embeddings for specific purposes...
        embeddings=[doc.embedding],
        documents=[doc.full_text],
        metadatas=metas,
        ids=[uuid.uuid4().hex]
    )


where_or = lambda x: {"$or": x}


class AgentBase():
    # TODO: define BlogWritingAgent as a pipeline as well!!

    _context_where_all = where_or([{"information_type": "question"},
                                   {"information_type": "task"},
                                   {"information_type": "info"}])

    _context_where_info = where_or([{"information_type": "question"},
                                    {"information_type": "info"}])

    _context_where_tasks = {"information_type": "task"}

    def __init__(self, chromadb_collection, objective):
        self._chromadb_collection = chromadb_collection
        self._objective = objective
        self._debug_queue = []
        self._final_result = ""

    def add_question(self, question, answer):
        doc = Document(f"question: {question.strip()}\nanswer: {answer.strip()}")
        add_info_to_collection(
            self._chromadb_collection, doc,
            [{"information_type": "question", "question": question.strip(), "answer": answer.strip()}])

    def add_task(self, task, result):
        doc = Document(f"task: {task.strip()}\nresult: {result.strip()}")
        add_info_to_collection(
            self._chromadb_collection, doc,
            [{"information_type": "task", "task": task.strip(), "result": result.strip()}])

    def add_data(self, key, data):
        doc = Document(f"{key.strip()}: {data.strip()}")
        add_info_to_collection(
            collection, doc,
            [{"information_type": "data", "key": key.strip(), "info": data.strip()}])

    def task_chat(self, task, context=None, previous_tasks=None, format="yaml"):
        msgs = []
        msgs.append({"role": "system",
                     "content": "You are a helpful assistant that aims to complete one task."})
        msgs.append({"role": "user",
                     "content": f"# Overall objective: \n{self._objective}\n\n"}),
        if previous_tasks:
            msgs.append({"role": "user",
                         "content": f"# Take into account these previously completed tasks"
                                    f"\n\n{previous_tasks} \n\n"}),
        if context:
            msgs.append({"role": "user",
                         "content": f"# Take into account this context"
                                    f"\n\n{context} \n\n"}),
        msgs.append(
            {"role": "user", "content": f"# Complete the following task: \n{task} \n\n"
                                        f"Provide only the precise information requested without context, "
                                        f"make sure we can parse the response as {format}. RESULT:\n"})
        return msgs

    def get_context(self, task: str, n_results: int = 5, where_clause=None):
        where_clause = where_clause or self._context_where_all
        context = query(task, where=where_clause,
                        n_results=n_results)["documents"]
        return context

    def execute_task(
            self,
            task,
            context_size: int = 5,
            previous_task_size=0,
            max_tokens=256,
            format="yaml"
    ):
        if context_size:
            context = self.get_context(task,
                                       where_clause=self._context_where_info,
                                       n_results=context_size)[0]
        else:
            context = ""
        if previous_task_size:
            previous_tasks = self.get_context(
                task, where_clause=self._context_where_info,
                n_results=previous_task_size)[0]
        else:
            previous_tasks = ""
        msgs = self.task_chat(previous_tasks=previous_tasks,
                              context="\n---\n".join(context), task=task,
                              format=format)
        msg = '\n'.join(m['content'] for m in msgs)
        logger.info(f"execute_task: {msg}")
        res = openai_chat_completion(msgs, max_tokens=max_tokens).content
        self._debug_queue.append((msgs, res))
        return res


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

    idx = ds.e('text_segments', meta_properties=["file_meta"])

    collection = client.get_or_create_collection(name="index")
    compute, query = idx.add_to_chroma(collection)  # remove number to calculate for all files!
    # create the index if we run it for the first time!
    need_index = False if collection.count() > 0 else True
    if need_index:
        # collection = client.get_collection(name="index")
        client.delete_collection(name="index")
        collection = client.get_or_create_collection(name="index")
        with ProgressBar():
            # idx.idx_dict.take(200, npartitions=3)  # remove number to calculate for all files!
            # idx.idx_dict.compute()
            compute()

    # make sure we save the index on harddisk!
    client.persist()

    ######  Start the writing process  #####

    final_result = []

    # TODO: do this in a cli..
    agent = AgentBase(
        chromadb_collection=collection,
        objective="Write a blog post, introducing a new library (which was developed by us, "
                  "the company 'Xyntopia') to " \
                  "visitors of our corporate webpage, which might want to use the pydoxtools library but "
                  "have no idea about programming. Make sure, the text is about half a page long."
    )

    # first, add a basic answer, to get the algorithm startet more quickly :).
    agent.add_question(question="Can you please provide the main topic of the project or some primary " \
                                "keywords related to the project, " \
                                "to help with identifying the relevant files in the directory?",
                       answer="python library, AI, pipelines")

    task = "What additional information do you need to create a first, very short outline as a draft? " \
           "provide it as a ranked list of questions"
    res = agent.execute_task(task)
    questions = yaml_loader(res)[:5]
    agent.add_task(task, str(questions))

    for question in questions:
        task = "Produce a list of 3-5 word strings which we can convert " \
               "to embeddings and then use those for a " \
               f"nearest neighbour search. We need them to be similar to text snippets which are" \
               f"capable of addressing the question: '{question}'"
        res = agent.execute_task(task=task)
        list_of_search_strings = yaml.unsafe_load(res)
        agent.add_task(task, str(list_of_search_strings))

        # take the mean of all the above search strings :)
        embedding = np.mean([Document(q).embedding for q in list_of_search_strings], 0).tolist()
        # make sure we have no duplicate results
        res = query(embeddings=embedding, where={"document_type": "text/markdown"})
        txt = "\n\n".join(list(set(res["documents"][0])))

        # sometimes we can answer a question only with this
        # TODO: try to formulate questions in a way that we can answer them with
        #       only a single word...
        anslist = pd.DataFrame(Document(txt).answers(question)[0])
        # if not anslist.empty:
        #    ans = anslist.groupby(0).sum().sort_values(by=1, ascending=False).index[0]
        # else:
        res = agent.execute_task(task=f"answer the following question: '{question}' "
                                      f"using this text as input: {txt}")
        ans = res
        # TODO: ask questions like "is this correct?" or can you provide the name?
        # ans_parsed = yaml.unsafe_load(ans)
        agent.add_question(question, ans)

    long_text = False  # longer than certain max_tokens
    if long_text:
        task = "Create an outline for our objective and format it as a flat yaml dictionary," \
               "indicating the desired number of words for each section."
        res = agent.execute_task(task)
        outline = yaml_loader(res)
        agent.add_task(task, res)

        sections = []
        for section_name, word_num in outline.items():
            task = f"Create a list of keywords for the section: " \
                   f"{section_name} with approx {word_num} words"
            res = agent.execute_task(task)
            section_keywords = yaml_loader(res)

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
            section_text = yaml_loader(res)

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
    critique = yaml_loader(res)

    task = "Given this text:\n\n" \
           f"```markdown\n{txt}\n```\n\n" \
           f"and its critique: {critique}\n\n" \
           "Generate instructions " \
           "that would make it better. Sort them by importance and return" \
           " it as a list of tasks"
    res = agent.execute_task(task, context_size=0, max_tokens=1000)
    tasks = yaml_loader(res)

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
