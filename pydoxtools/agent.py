import uuid

import numpy as np
import yaml

from pydoxtools import Document
from pydoxtools.extract_nlpchat import openai_chat_completion


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


class AgentBase:
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
        embedding = np.mean([Document(q).embedding for q in list_of_search_strings], 0).tolist()
        # make sure we have no duplicate results
        # TODO: make sure we are searching more documents than just this one...
        res = query(embeddings=embedding, where={"document_type": "text/markdown"})
        txt = "\n\n".join(list(set(res["documents"][0])))

        # sometimes we can answer a question only with this
        # TODO: try to formulate questions in a way that we can answer them with
        #       only a single word...
        #       also, try this "local" strategy to answer questions...
        # anslist = pd.DataFrame(Document(txt).answers(question)[0])
        # if not anslist.empty:
        #    ans = anslist.groupby(0).sum().sort_values(by=1, ascending=False).index[0]
        # else:
        res = agent.execute_task(task=f"answer the following question: '{question}' "
                                      f"using this text as input: {txt}")
        ans = res
        # TODO: ask questions like "is this correct?" or can you provide the name?
        # ans_parsed = yaml.unsafe_load(ans)
        self.add_question(question, ans)
        return ans
