import functools
import logging
import uuid

import chromadb
import dask.diagnostics
import numpy as np
import yaml

import pydoxtools as pdx
from pydoxtools.extract_nlpchat import openai_chat_completion

logger = logging.getLogger(__name__)


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


def add_info_to_collection(collection, doc: pdx.Document, metas: list[dict]):
    collection.add(
        # TODO: embeddings=  #use our own embeddings for specific purposes...
        embeddings=[float(v) for v in doc.embedding],
        # embeddings=[copy.copy(doc.embedding.tolist())],
        documents=[doc.full_text],
        metadatas=metas,
        ids=[uuid.uuid4().hex]
    )


where_or = lambda x: {"$or": x}


class AgentBase:
    # TODO: define BlogWritingAgent as a pipeline as well?
    #       not sure, yet, if we would benefit from this here...
    #       might make sense in the future, if we have multiple
    #       agents for different tasks...

    _context_where_all = where_or([{"information_type": "question"},
                                   {"information_type": "task"},
                                   {"information_type": "info"}])

    _context_where_info = where_or([{"information_type": "question"},
                                    {"information_type": "info"}])

    _context_where_tasks = {"information_type": "task"}

    def __init__(
            self,
            objective: str,
            # TODO: support multiple datasources
            data_source: pdx.DocumentBag,
            vector_store: chromadb.config.Settings,
            startfresh: bool = True,
            privacy_mode="non_proviate_llm_queries"
    ):
        if not isinstance(vector_store, chromadb.config.Settings):
            raise NotImplementedError("Other vectorstores besides chromadb are WIP!")
        if privacy_mode != "non_proviate_llm_queries":
            raise RuntimeError("You have to have 'Alpaca' model installed in order"
                               "to run other privacy modes...")
        self.vector_store = vector_store
        self._objective = objective
        self._debug_queue = []
        self._final_result = ""
        self._data_source: pdx.DocumentBag | None = data_source
        # we need to delete all previously stored information
        # in most cases to make our information search more efficient and deduplicate information
        # TODO: make agents reuse information in the future!
        if startfresh:
            self.chromadb_collection.delete(where=self._context_where_all)

    @property
    def collection_name(self):
        return "pdx_agent_index"

    def reload_vector_store(self):
        """sometimes we need to reload the vector store. For example if we have added nwe data from
        a different thread..."""
        del self.chromadb_client
        del self.chromadb_collection

    @functools.cached_property
    def chromadb_client(self):
        client = chromadb.Client(self.vector_store)
        return client

    @functools.cached_property
    def chromadb_collection(self):
        info_collection = self.chromadb_client.get_or_create_collection(name=self.collection_name)
        return info_collection

    @property
    def vectorize(self):
        return self._data_source.vectorizer

    @property
    def documents(self):
        return self._data_source

    def add_question(self, question, answer):
        doc = self.documents.Document(f"question: {question.strip()}\nanswer: {answer.strip()}")
        add_info_to_collection(
            self.chromadb_collection, doc,
            [{"information_type": "question", "question": question.strip(), "answer": answer.strip()}])

    def add_task(self, task, result):
        doc = self.documents.Document(f"task: {task.strip()}\nresult: {result.strip()}")
        add_info_to_collection(
            self.chromadb_collection, doc,
            [{"information_type": "task", "task": task.strip(), "result": result.strip()}])

    def add_data(self, key, data):
        doc = self.documents.Document(f"{key.strip()}: {data.strip()}")
        add_info_to_collection(
            self.chromadb_collection, doc,
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
        context = self.chromadb_collection.query(
            query_embeddings=[self.vectorize(task).tolist()],
            where=where_clause,
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

    def research_question(self, question):
        task = "Produce a list of 3-5 word strings which we can convert " \
               "to embeddings and then use those for a " \
               f"nearest neighbour search. We need them to be similar to text snippets which are" \
               f"capable of addressing the question: '{question}'"
        res = self.execute_task(task=task)
        list_of_search_strings = yaml.unsafe_load(res)
        self.add_task(task, str(list_of_search_strings))

        # TODO: think about how we can integrate this below with Document...
        # take the mean of all the above search strings :)
        embedding: list[float] = np.mean([self.vectorize(q) for q in list_of_search_strings], 0).tolist()
        # make sure we have no duplicate results
        # TODO: make sure we are searching more documents than just this one...
        # TODO: let the AI choose which document types we will search through to answer
        #       this question...
        res = self.chromadb_collection.query(
            query_embeddings=[embedding],
            where={"document_type": "text/markdown"},
            n_results=5)
        txt = "\n\n".join(list(set(res["documents"][0])))

        # sometimes we can answer a question only with this
        # TODO: try to formulate questions in a way that we can answer them with
        #       only a single word...
        #       also, try this "local" strategy to answer questions...
        # anslist = pd.DataFrame(Document(txt).answers(question)[0])
        # if not anslist.empty:
        #    ans = anslist.groupby(0).sum().sort_values(by=1, ascending=False).index[0]
        # else:
        res = self.execute_task(task=f"answer the following question: '{question}' "
                                     f"using this text as input: {txt}")
        ans = res
        # TODO: ask questions like "is this correct?" or can you provide the name?
        # ans_parsed = yaml.unsafe_load(ans)
        self.add_question(question, ans)
        return ans

    def pre_compute_index(self):
        """Create an index from our datasource by splitting it up into !"""
        # TODO: use "generalized" metadata..   to make sure we can also
        #       use information stored in databases for example...
        idx = self._data_source.apply(new_document='text_segments', document_metas="file_meta")
        idx = idx.exploded
        compute, _ = idx.add_to_chroma(self.vector_store, self.collection_name)

        # create the index if we run it for the first time!
        # collection = client.get_collection(name="index")
        self.chromadb_client.delete_collection(name=self.collection_name)
        self.chromadb_client.persist()
        with dask.diagnostics.ProgressBar():
            # idx.idx_dict.take(200, npartitions=3)  # remove number to calculate for all files!
            # idx.idx_dict.compute()
            compute()

        self.reload_vector_store()
