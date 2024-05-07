

"""Run inference."""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy

from manifest import Manifest

import utils.data_utils as data_utils
import utils.prompt_utils as prompt_utils
from utils import constants
from utils.utils import compute_metrics, setup_logger

import requests
import json
import time


def call_kalliope(prompt, max_tokens):
    headers = {
        'Authorization': 'glpat-6RMEnrwgPovfC4gWLXxc',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'llama-2-70b-chat.q5_k_m',
        'max_tokens': max_tokens,
        'messages': [
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    }

    response = requests.post('https://api.kalliope.bigtwitter.cloud.edu.au/v1/chat/completions', headers=headers, json=json_data)
    #print(response.content)
    choices = json.loads(response.content)["choices"]
    result = choices[0]
    if(result['message']['content'][0] == " "):
        return result['message']['content'][1:]
    else:
        return result['message']['content']



'''
message = json.loads(result.content)["message"]
print(message)'''
# print(type(int(json.loads(response.content)["result"])))  # 取出响应内容中 "result" 字段的值
# print(int(json.loads(response.content)["result"], 16))  # 16 进制字符串转成整型 int
# print(type(int(json.loads(response.content)["result"], 16)))
# Note: json_data will not be serialized by requests
# exactly as it was in the original request.
# data = '{ "model":"llama-2-70b-chat.q5_k_m", "max_tokens":100, "messages":[{"role": "user", "content": "Product A is title: da-lite da-glas deluxe rear projection screen - 50 x 50 av format. modelno: 27647. Product B is title: da-lite 27658 da-glas deluxe rear projection screen - 57 3 4 x 77 video format. modelno: nan. Are A and B the Same?    "}] }'
# response = requests.post('https://api.kalliope.bigtwitter.cloud.edu.au/v1/chat/completions', headers=headers, data=data)

logger = logging.getLogger(__name__)
Budget = 10
result = []
batchsize = 3
def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        default="D:/GitHub/LLM-EM0001/fm_data_tasks-main/fm_data_tasks/data/datasets/entity_matching/structured/Walmart-Amazon",
        ##todo: It is possible to change the default --data_dir above
        ##required=True,
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        help="Which data directory to run.",
        default="outputs/Walmart-Amazon/validation/default/7k_0inst_0cb_random_200run_0dry/trial_0.feather",
        ##todo: It is possible to change the default --data_dir above
        ##required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument(
        "--cache_name",
        type=str,
        help="Manifest cache type.",
        default="sqlite",
        choices=["redis", "sqlite", "noop"],
    )
    parser.add_argument(
        "--cache_connection",
        type=str,
        help="Manifest cache connection string.",
        default="fm_data_tasks.sqlite",
    )
    parser.add_argument(
        "--client_name",
        type=str,
        help="Manifest client type.",
        default="huggingface",
        choices=["openai", "opt", "huggingface", "Kalliope"],
    )
    parser.add_argument(
        "--client_connection",
        type=str,
        help="Manifest client connection string.",
        #chatgpt
        default="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",
        # huggingface(to be confirmed)
        #default="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",
        # Kalliope
        # default="glpat-6RMEnrwgPovfC4gWLXxc"
        ##todo: It is possible to change the default --client_connection above
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        help="Tag for run saving.",
        default="default",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite sqlite cache of input/output results.",
    )
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=2)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        ##todo: When using validation_clusters, the error "PermissionError: [Errno 13] Permission denied: 'D:/GitHub/LLM-EM0001/fm_data_tasks-main/fm_data_tasks/data/datasets/entity_matching/structured/Walmart-Amazon'" happens, why?
        choices=["random", "manual", "validation_clusters", "SSB"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks \
             with random prompts.",
        action="store_true",
    )
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    parser.add_argument(
        "--nan_tok",
        type=str,
        help="Token to represent nan entries. Default is 'nan'.",
        default="nan",
    )
    parser.add_argument(
        "--num_run",
        type=int,
        help="Number examples to run through model.",
        default=200,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number trials to run. Results will be averaged with variance reported.",
        default=5,
    )
    parser.add_argument(
        "--num_print",
        type=int,
        help="Number example prompts to print.",
        default=200,
    )
    parser.add_argument(
        "--add_task_instruction",
        help="Add task instruction to the prompt before examples.",
        action="store_true",
    )
    parser.add_argument("--task_instruction_idx", type=int, default=0)
    parser.add_argument("--do_test", help="Run on test file.", action="store_true")
    parser.add_argument(
        "--dry_run", help="Dry run. Do not actually ping model.", action="store_true"
    )

    parser.add_argument(
        "--stop_token", help="Token to stop on for a given generated response", default="\n"
    )

    # Model args
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to generate.", default=3
    )

    args = parser.parse_args()
    return args

def main():
    """Run main method."""

    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("num_trials must be greater than 0.")
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))

    # Will set seed for pandas
    numpy.random.seed(args.seed)

    test_file = "test" if args.do_test else "validation"

    # Read pandas DF datasets
    pd_data_files = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_instruction=False,
        max_train_samples=-1,
        max_train_percent=-1,
        sep_tok=args.sep_tok,
        nan_tok=args.nan_tok,
    )
    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")

    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    task = constants.DATA2TASK[args.data_dir]
    logger.info(f"Using {args.task_instruction_idx} instruction idx")
    task_instruction = constants.DATA2INSTRUCT[args.data_dir]
    num_run = args.num_run
    if args.num_run == -1:
        num_run = test_data.shape[0]
    num_run = min(num_run, test_data.shape[0])

    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")
    logger.info(f"Running {num_run} examples for {args.num_trials} trials.")

    # Setup manifest
    manifest = Manifest(
        cache_name=args.cache_name,
        cache_connection=args.cache_connection,
        client_name=args.client_name,
        client_connection=args.client_connection,
        stop_token=args.stop_token,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
        n=1,
    )
    if args.add_task_instruction:
        prompt = lambda x: f"{task_instruction} {x}"
    else:
        prompt = lambda x: f"{x}"
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    saved_prefix = None
    for trial_num in range(args.num_trials):
        numpy.random.seed(args.seed + trial_num)
        queries = []
        for _, row in test_data.iterrows():
            serialized_r = row["text"]
            if args.sample_method == "manual":
                prefix_exs = prompt_utils.get_manual_prompt(args.data_dir, row)
            elif args.sample_method == "validation_clusters":
                if saved_prefix is None:
                    logger.info("Generating validation cluster prompt.")
                    saved_prefix = prompt_utils.get_validation_prompt(
                        args.validation_path,
                        num_examples=args.k,
                        task=task,
                    )
                prefix_exs = saved_prefix
            elif args.sample_method == "SSB":
                if saved_prefix is None:
                    saved_prefix = prompt_utils.stratified_shapley_bandits_prompt(
                        pd_data_files["train"], num_examples=args.k,budget = Budget
                    )
                prefix_exs = saved_prefix
            else:##here sample_method = random
                if saved_prefix is None:
                    saved_prefix = prompt_utils.get_random_prompt(
                        pd_data_files["train"], num_examples=args.k
                    )
                prefix_exs = saved_prefix
            queries.append((prefix_exs + "\n" + serialized_r).strip())

        gt = test_data["label_str"]
        preds = []
        idx = 0
        timelist = []
        totaltime = 0
        start_time = time.time()
        # Run a few for printing -- they are cached
        for _ in range(min(num_run, args.num_print)):
            logger.info(prompt(queries[idx]))
            if not args.dry_run:
                '''pred = manifest.run(
                    prompt(queries[idx]), overwrite_cache=args.overwrite_cache
                )'''
                pred = call_kalliope(prompt(queries[idx]), args.max_tokens)
            else:
                pred = ""
            preds.append(pred)
            result.append(pred)
            print(result)
            logger.info(f"====> {pred} <====")
            idx += 1
            end_time = time.time()
            totaltime += end_time - start_time
            timelist.append((idx, end_time-start_time))
            #print("running time:", end_time - start_time)

        # Save trial predictions
        save_data = test_data.iloc[:num_run].copy(deep=True).reset_index()
        gt = gt[:num_run]
        save_data["preds"] = preds
        save_data["queries"] = queries[:num_run]

        prec, rec, acc, f1 = compute_metrics(preds, gt, task)

        logger.info(
            f"Metrics Trial {trial_num}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        output_file = (
            Path(args.output_dir)
            / f"{Path(args.data_dir).stem}"
            / f"{test_file}"
            / f"{args.run_tag}"
            / f"{args.k}k"
            f"_{int(args.add_task_instruction)}inst"
            f"_{int(args.class_balanced)}cb"
            f"_{args.sample_method}"
            f"_{args.num_run}run"
            f"_{int(args.dry_run)}dry" / f"trial_{trial_num}.feather"
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved to {output_file}")

        save_data.to_feather(output_file)

    for k, values in list(trial_metrics.items()):
        trial_metrics[f"{k}_avg"] = numpy.average(values)
        trial_metrics[f"{k}_std"] = numpy.std(values)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))

    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")
    print("avgtime", totaltime/200)
    print("timelist", timelist)


if __name__ == "__main__":
    main()