import os
import json
import time
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core.schema import NodeWithScore, ImageNode

from search_engine import SearchEngine,HybridSearchEngine

from llms.llm import LLM
from llms.evaluator import Evaluator

from utils.overall_evaluator import eval_search,eval_search_type_wise
from vidorag_agents import ViDoRAG_Agents

class MMRAG:
    def __init__(self,
                dataset='ExampleDataset',
                query_file='rag_dataset.json',
                experiment_type = 'retrieval_infer',
                generate_vlm='qwen-vl-max',
                embed_model_name='BAAI/bge-m3',
                embed_model_name_vl=None, # openbmb/VisRAG-Ret vidore/colqwen2-v1.0
                embed_model_name_text=None, # nvidia/NV-Embed-v2 BAAI/bge-m3
                workers_num = 1,
                topk=10,
                vllm_base_url=None,
                vllm_api_key=None,
                output_file=None,
                result_order=None):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.is_vidoseek = Path(dataset).name.lower() == "vidoseek"
        self.query_file = query_file
        self.generate_vlm = generate_vlm
        self.vllm_base_url = vllm_base_url
        self.vllm_api_key = vllm_api_key
        self.result_order = result_order
        self.dataset_dir = os.path.join('./data', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.vlm = None
        # self.evaluator = Evaluator()
        
        # load search_engine
        if embed_model_name_vl is not None and embed_model_name_text is not None:
            self.search_engine = HybridSearchEngine(self.dataset,
                                                    embed_model_name_vl=embed_model_name_vl,
                                                    embed_model_name_text=embed_model_name_text,
                                                    topk=topk)
        else:
            self.search_engine = SearchEngine(self.dataset, embed_model_name=embed_model_name)

        # retrieval only
        if experiment_type == 'retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.output_file_name = f'base_retrieval_{embed_model_name}.jsonl'
        # hybrid retrieval
        elif experiment_type == 'dynamic_hybird_retrieval_infer':
            self.eval_func = self.retrieval_infer
            self.search_engine.gmm = True
            self.output_file_name = f'dynamic_hybird_retrieval_{embed_model_name_vl}_{embed_model_name_text}.jsonl'
        # vidorag
        elif experiment_type == 'vidorag':
            if not generate_vlm.startswith('gpt') and (vllm_base_url is None or vllm_api_key is None):
                raise ValueError("--vllm_base_url and --vllm_api_key are required when --generate_vlm is not a gpt model.")
            self.vlm = LLM(model_name=generate_vlm, base_url=vllm_base_url, api_key=vllm_api_key)
            self.agents = ViDoRAG_Agents(self.vlm)
            self.eval_func = self.vidorag
            self.search_engine.gmm = True
            self.search_engine.gmm_candidate_length = True
            self.output_file_name = f'vidorag_{generate_vlm}.json'
        
        if output_file is None:
            self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))
        else:
            self.output_file_path = output_file
        Path(self.output_file_path).parent.mkdir(parents=True, exist_ok=True)

    def retrieval_infer(self,sample):
        query = sample['query']
        recall_results = self.search_engine.search(query)
        recall_results['source_nodes'] = recall_results['source_nodes'][:100]
        sample['recall_results'] = recall_results
        return sample
    def vidorag(self,sample):
        started_at = time.perf_counter()
        self.vlm.reset_usage()
        query = sample['query']
        print(query)
        recall_results = self.search_engine.search(query)
        candidate_image = [
            self._candidate_image_path(node)
            for node in recall_results['source_nodes']
        ]
        if 'gmm' not in self.experiment_type:
            candidate_image = candidate_image[:self.top_k]
        try:
            answer, trace = self.agents.run_agent(
                query,
                candidate_image,
                return_trace=True,
            )
        except Exception as e:
            print(e)
            raise e
            return None

        return {
            "query": query,
            "answer": answer,
            "retrieved_corpus_ids": [
                self._corpus_id_from_image(image) for image in candidate_image
            ],
            "gt_corpus_ids": self._ground_truth_corpus_ids(sample),
            "token_usage": self.vlm.get_usage(),
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "trace": self._format_trace(trace),
        }

    def _query_file_path(self):
        query_path = Path(self.query_file)
        if query_path.is_file():
            return query_path
        return Path(self.dataset_dir) / query_path

    def _load_query_data(self):
        query_path = self._query_file_path()
        with query_path.open("r", encoding="utf-8") as f:
            if query_path.suffix == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)["examples"]

        if query_path.suffix == ".jsonl" and not self.is_vidoseek:
            doc_name = Path(self.dataset).name
            data = [
                sample for sample in data
                if sample.get("doc_name") == doc_name
            ]
            if not data:
                raise ValueError(f"{query_path}: no questions found for document {doc_name}")

        seen_ids = set()
        for index, sample in enumerate(data):
            sample_id = sample.get("id", sample.get("uid"))
            if sample_id is None:
                raise ValueError(f"{query_path}: item {index} has no id or uid")
            sample_key = str(sample_id)
            if sample_key in seen_ids:
                raise ValueError(f"{query_path}: duplicate question id {sample_key}")
            if "query" not in sample:
                raise ValueError(f"{query_path}: question {sample_key} has no query")
            seen_ids.add(sample_key)

        return data

    def _candidate_image_path(self, node):
        metadata = node["node"]["metadata"]
        source_name = metadata.get("file_name", metadata.get("filename"))
        if source_name is None:
            raise ValueError("Retrieved node has no file_name or filename metadata")
        source_path = Path(source_name)
        candidates = []
        if source_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            candidates.append(Path(self.img_dir) / source_path.name)
        candidates.extend(
            Path(self.img_dir) / f"{source_path.stem}{suffix}"
            for suffix in (".png", ".jpg", ".jpeg")
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"No image found for retrieved node {source_name} in {self.img_dir}"
        )

    def _corpus_id_from_image(self, image):
        stem = Path(image).stem
        if self.is_vidoseek:
            match = re.match(r"^.+_(\d+)_(\d+)$", stem)
            if match is not None:
                return int(match.group(2))
        else:
            match = re.match(r"^page_(\d+)$", stem)
            if match is not None:
                return int(match.group(1))
        raise ValueError(
            f"Unexpected {'ViDoSeek' if self.is_vidoseek else 'MMDocRAG'} "
            f"image name: {Path(image).name}"
        )

    def _format_trace(self, trace):
        formatted_trace = []
        for entry in trace:
            formatted_entry = dict(entry)
            for key in list(formatted_entry):
                if key.endswith("_images"):
                    images = formatted_entry.pop(key)
                    formatted_entry[key.replace("_images", "_corpus_ids")] = [
                        self._corpus_id_from_image(image) for image in images
                    ]
            formatted_trace.append(formatted_entry)
        return formatted_trace

    @staticmethod
    def _ground_truth_corpus_ids(sample):
        page = sample.get("page")
        if isinstance(page, list):
            return [int(value) for value in page]
        if isinstance(page, int):
            return [page]
        reference_page = sample.get("meta_info", {}).get("reference_page")
        if isinstance(reference_page, list):
            return [int(value) for value in reference_page]
        raise ValueError(f"Question {sample.get('id', sample.get('uid'))} has no valid page")

    def _write_vidorag_results(self, results, ordered_ids):
        ordered_results = {
            sample_id: results[sample_id]
            for sample_id in ordered_ids
            if sample_id in results
        }
        output_path = Path(self.output_file_path)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(ordered_results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temporary_path, output_path)

    def _vidorag_failure_result(self, item, error):
        try:
            token_usage = self.vlm.get_usage()
        except Exception:
            token_usage = {}
        try:
            gt_corpus_ids = self._ground_truth_corpus_ids(item)
        except Exception:
            gt_corpus_ids = []
        return {
            "query": item.get("query"),
            "answer": None,
            "retrieved_corpus_ids": [],
            "gt_corpus_ids": gt_corpus_ids,
            "token_usage": token_usage,
            "latency_seconds": None,
            "trace": [],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    def _eval_vidorag_dataset(self, data):
        ordered_ids = self.result_order or [
            str(item.get("id", item.get("uid"))) for item in data
        ]
        results = {}
        if os.path.exists(self.output_file_path):
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not isinstance(results, dict):
                raise ValueError(f"{self.output_file_path} must contain a JSON object")

        pending = [
            item for item, sample_id in zip(data, ordered_ids)
            if sample_id not in results
        ]
        if self.workers_num == 1:
            for item in tqdm(pending):
                try:
                    result = self.vidorag(item)
                except Exception as error:
                    print(
                        f"ViDoRAG sample {item.get('id', item.get('uid'))} failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    result = self._vidorag_failure_result(item, error)
                if result is None:
                    continue
                results[str(item.get("id", item.get("uid")))] = result
                self._write_vidorag_results(results, ordered_ids)
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                future_to_item = {
                    executor.submit(self.vidorag, item): item
                    for item in pending
                }
                unsaved = 0
                for future in tqdm(
                    as_completed(future_to_item),
                    total=len(future_to_item),
                    desc="Processing",
                ):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        print(
                            f"ViDoRAG sample {item.get('id', item.get('uid'))} failed: "
                            f"{type(error).__name__}: {error}"
                        )
                        result = self._vidorag_failure_result(item, error)
                    if result is None:
                        continue
                    results[str(item.get("id", item.get("uid")))] = result
                    unsaved += 1
                    if unsaved >= 3:
                        self._write_vidorag_results(results, ordered_ids)
                        unsaved = 0
                if unsaved:
                    self._write_vidorag_results(results, ordered_ids)
        return self.output_file_path


    def eval_dataset(self):
        eval_func = self.eval_func

        data = self._load_query_data()
        if self.experiment_type == "vidorag":
            return self._eval_vidorag_dataset(data)
        
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            uid_already = [item['uid'] for item in results]
            data = [item for item in data if item['uid'] not in uid_already]
            
        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a") as f:
                    json.dump(result, f,ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    result = future.result()
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res,ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res,ensure_ascii=False) + "\n")

        return self.output_file_path

    def eval_overall(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search(data)
        with open(self.output_file_path.replace(".jsonl", "_eval.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
    
    def eval_overall_type_wise(self):
        data = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        results = eval_search_type_wise(data)
        with open(self.output_file_path.replace(".jsonl", "_eval_type_wise.json"), "w") as f:
            json.dump(results, f,indent=2,ensure_ascii=False)
        
def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ExampleDataset', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='retrieval_infer', help="The type of experiment")
    parser.add_argument("--embed_model_name", type=str, default='BAAI/bge-m3', help="The name of embedding model")
    parser.add_argument("--workers_num", type=int, default=1, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    parser.add_argument("--embed_model_name_vl", type=str, default=None, help="The name of embedding model for vl")
    parser.add_argument("--embed_model_name_text", type=str, default=None, help="The name of embedding model for text")
    parser.add_argument("--generate_vlm", type=str, default='qwen-vl-max', help="The name of VLM model")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1", help="The OpenAI-compatible vLLM base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="The vLLM API key. Use EMPTY if your vLLM server does not require authentication")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON path for vidorag")
    parser.add_argument("--all_documents", action="store_true", help="Run every MMDocRAG doc_name in the query JSONL")
    args = parser.parse_args()
    return args

def build_mmrag(args, dataset, output_file=None, result_order=None):
    return MMRAG(
        dataset=dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        embed_model_name=args.embed_model_name,
        workers_num=args.workers_num,
        topk=args.topk,
        embed_model_name_vl=args.embed_model_name_vl,
        embed_model_name_text=args.embed_model_name_text,
        generate_vlm=args.generate_vlm,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        output_file=output_file,
        result_order=result_order,
    )

def run_all_documents(args):
    if args.experiment_type != "vidorag":
        raise ValueError("--all_documents currently supports only --experiment_type vidorag")

    query_path = Path(args.query_file)
    if not query_path.is_file() or query_path.suffix != ".jsonl":
        raise ValueError("--all_documents requires --query_file to be an existing JSONL path")
    with query_path.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if not questions or any("doc_name" not in item for item in questions):
        raise ValueError(f"{query_path} must contain MMDocRAG questions with doc_name")
    if any(not isinstance(item.get("page"), list) for item in questions):
        raise ValueError(
            "--all_documents expects MMDocRAG questions whose page field is a list"
        )

    doc_names = list(dict.fromkeys(item["doc_name"] for item in questions))
    ordered_ids = [str(item.get("id", item.get("uid"))) for item in questions]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError(f"{query_path} contains duplicate question IDs")

    output_file = args.output_file
    if output_file is None:
        model_name = args.generate_vlm.replace("/", "-")
        output_file = os.path.join(
            "data",
            "results",
            f"mmdocrag_vidorag_{model_name}.json",
        )

    completed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        if not isinstance(existing_results, dict):
            raise ValueError(f"{output_file} must contain a JSON object")
        completed_ids = set(existing_results)

    for index, doc_name in enumerate(doc_names, start=1):
        doc_question_ids = {
            str(item.get("id", item.get("uid")))
            for item in questions
            if item["doc_name"] == doc_name
        }
        if doc_question_ids <= completed_ids:
            print(f"[{index}/{len(doc_names)}] Skipping completed document: {doc_name}")
            continue
        dataset_dir = Path("data") / doc_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"MMDocRAG dataset directory not found: {dataset_dir}")

        print(f"[{index}/{len(doc_names)}] Processing document: {doc_name}")
        mmrag = build_mmrag(
            args,
            dataset=doc_name,
            output_file=output_file,
            result_order=ordered_ids,
        )
        mmrag.eval_dataset()
        completed_ids.update(doc_question_ids)
        del mmrag

    return output_file

if __name__ == "__main__":
    args = arg_parse()
    if args.all_documents:
        run_all_documents(args)
        raise SystemExit(0)

    mmrag = build_mmrag(
        args,
        dataset=args.dataset,
        output_file=args.output_file,
    )
    mmrag.eval_dataset()
    if args.experiment_type != "vidorag":
        mmrag.eval_overall()
        mmrag.eval_overall_type_wise()
