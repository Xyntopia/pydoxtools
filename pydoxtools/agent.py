from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import logging
import typing
import uuid

import chromadb
import dask.diagnostics
import numpy as np

import pydoxtools as pdx
import pydoxtools.extract_nlpchat

logger = logging.getLogger(__name__)


def generate_function_usage_prompt(func: callable):
    # TODO: summarize function usage, only extract the parameter description
    # func.__doc__
    return func.__doc__


def add_info_to_collection(collection, doc: pdx.Document, metas: list[dict]):
    collection.add(
        # TODO: embeddings=  #use our own embeddings for specific purposes...
        embeddings=[float(v) for v in doc.embedding],
        # embeddings=[copy.copy(doc.embedding.tolist())],
        documents=[doc.full_text],
        metadatas=metas,
        ids=[uuid.uuid4().hex]
    )


def where_or(x):
    if len(x) > 1:
        return {"$or": x}
    elif len(x) == 1:
        return x[0]


class LLMAgent:
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
            privacy_mode="non_private_llm_queries"
    ):
        if not isinstance(vector_store, chromadb.config.Settings):
            raise NotImplementedError("Other vectorstores besides chromadb are WIP!")
        if privacy_mode != "non_private_llm_queries":
            raise RuntimeError("You have to have 'Alpaca' model installed in order"
                               "to run other privacy modes...")
        self.vector_store = vector_store
        self._objective = objective
        self._debug_queue = []
        self._final_result = ""
        self._data_source: pdx.DocumentBag | None = data_source
        # we need to delete all previously stored "dynamic" information generated
        # by the agent in most cases to make our information
        # search more efficient and deduplicate information
        # TODO: make agents reuse information in the future!
        if startfresh:
            self.chromadb_collection.delete(where=self._context_where_all)
            self.chromadb_client.persist()

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

    def task_chat(self, task, context=None, previous_tasks=None, format="yaml", method="prompt"):
        pydoxtools.extract_nlpchat.task_chat(**locals())

    def get_context(self, task: str, n_results: int = 5, where_clause=None):
        where_clause = where_clause or self._context_where_all
        context = self.chromadb_collection.query(
            query_embeddings=[self.vectorize(task).tolist()],
            where=where_clause,
            n_results=n_results)["documents"]
        return context

    def get_data(self, where_clause):
        context = self.chromadb_collection.get(
            where=where_clause)
        return context

    def execute_task(
            self,
            task,
            context_size: int = 5,
            previous_task_size=0,
            max_tokens=256,
            formatting="yaml",
            save_task=False
    ) -> typing.Any:
        if context_size:
            context = self.get_context(
                task,
                where_clause=self._context_where_info,
                n_results=context_size)[0]
        else:
            context = ""
        # TODO: if we had this exact task in an earlier iteration..
        #       evaluate if we have to do it again or already have our answer...
        if previous_task_size:
            previous_tasks = self.get_context(
                task, where_clause=self._context_where_tasks,
                n_results=previous_task_size)[0]
        else:
            previous_tasks = ""
        res, msgs = pydoxtools.extract_nlpchat.execute_task(
            task=task, previous_tasks=previous_tasks, context=context, formatting=formatting,
            max_tokens=max_tokens, objective=self._objective)
        msg = '\n'.join(m['content'] for m in msgs)
        logger.info(f"execute_task: {msg}")
        self._debug_queue.append((msgs, res))
        if save_task:
            self.add_task(task, str(res))
        return res

    def research_questions(self, questions, allowed_documents: list[str] | None = None):
        for question in questions:
            self.research_question(question, allowed_documents=allowed_documents)

    def research_question(self, question, allowed_documents: list[str] | None = None):
        task = "Produce a list of 3-5 word strings which we can convert " \
               "to embeddings and then use those for a " \
               f"nearest neighbour search. We need them to be similar to text snippets which are" \
               f"capable of addressing the question: '{question}'"
        list_of_search_strings = self.execute_task(task=task, formatting="yaml")
        self.add_task(task, str(list_of_search_strings))

        # TODO: think about how we can integrate this below with Document...
        # take the mean of all the above search strings :)
        embedding: list[float] = np.mean([self.vectorize(q) for q in list_of_search_strings], 0).tolist()
        # make sure we have no duplicate results
        # TODO: make sure we are searching more documents than just this one...
        # TODO: let the AI choose which document types we will search through to answer
        #       this question...
        if allowed_documents:
            where = where_or([{"document_type": dtype} for dtype in allowed_documents])
        else:
            where = None
        res = self.chromadb_collection.query(
            query_embeddings=[embedding],
            where=where,
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
        # TODO: should we save the question?
        """If you don't know the answer, just say that you don't know, don't try to make up an answer."""
        res = self.execute_task(task=f"answer the following question: '{question}' "
                                     f"using this text as input: {txt}", formatting="txt")
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
