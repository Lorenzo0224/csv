"""Run inference."""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
import time
import heapq
import numpy
import pandas as pd
import csv
from manifest import Manifest

import utils.data_utils as data_utils
import utils.prompt_utils as prompt_utils
from utils import constants
from utils.utils import compute_metrics, setup_logger
import time
import requests
import pandas as pd
k = 2

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

##todo: Move these important global parameters into parse_args

logger = logging.getLogger(__name__)

result = []

def call_api(prefix_exs, prefix_exs2, shapley_value, target_index, size):
    v1 = call_api_once(prefix_exs)
    v2 = call_api_once(prefix_exs2)
    if(v1 == -100 or v2 == -100):
        return shapley_value, -100
    shapley_value[target_index][0] += v2-v1
    shapley_value[target_index][1] += 1
    return shapley_value, v2-v1

def call_api_once(prefix_exs, test = False, test_num = 100000):
    """Run main method."""
    """Generate args."""

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
    test_file = "test" if test else "validation"

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
    num_run = test_num if test else num_run
    if args.num_run == -1:
        num_run = test_data.shape[0]
    num_run = min(num_run, test_data.shape[0])
    #print(test_file, num_run)
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
    for trial_num in range(args.num_trials):
        numpy.random.seed(args.seed + trial_num)
        queries = []
        for _, row in test_data.iterrows():
            serialized_r = row["text"]

            queries.append((prefix_exs + "\n" + serialized_r).strip())

        gt = test_data["label_str"]
        preds = []
        idx = 0
        # Run a few for printing -- they are cached
        for _ in range(min(num_run, args.num_print)):
            logger.info(prompt(queries[idx]))
            pred = ""
            if not args.dry_run:
                '''pred = manifest.run(
                                   prompt(queries[idx]), overwrite_cache=args.overwrite_cache
                               )'''
                seconds = 0
                while(seconds < 240):
                    try:
                        pred = call_kalliope(prompt(queries[idx]), args.max_tokens)
                        break
                    except Exception as e:
                        print(e)
                        seconds += 20
                        time.sleep(seconds)
                        print(seconds)
                if(seconds == 240):
                    print("not responding after" + str(seconds) + "seconds")
                    return -100
                else:
                    print("reponded after" + str(seconds) + "seconds")
            preds.append(pred)
            result.append(pred)
            #print(result)
            logger.info(f"====> {pred} <====")
            #time.sleep(2)
            idx += 1

        # Send to model for predictions
        if not args.dry_run:
            for query in queries[idx:num_run]:
                #time.sleep(2)
                '''preds.append(
                    manifest.run(
                        prompt(query),
                        overwrite_cache=args.overwrite_cache,
                    )
                )'''
                preds.append(call_kalliope(prompt(query), args.max_tokens))
        else:
            preds.extend([""] * (num_run - idx))

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
    return trial_metrics["f1_avg"]

def exist_unsampled_data(SSB_sample_recorder):
        for j in range(len(SSB_sample_recorder)):
            if(sum(SSB_sample_recorder[j]) == 0):
                return j
        return -1

def initialize_percentage(SSB_sample_recorder):
    all = len(SSB_sample_recorder)
    cnt = 0
    for j in range(len(SSB_sample_recorder)):
        if (sum(SSB_sample_recorder[j]) != 0):
            cnt += 1
    return cnt/all

def target_random_prompt(train, sample_index):
    """Get random examples for prompt from train data."""
    '''first_index = sample_index[0]
    sel_train = train[first_index: first_index + 1]
    for index in sample_index[1:]:
        sel_train = pd.concat([train[index: index + 1], sel_train])'''
    sel_train = train.iloc[sample_index, :]
    serialized_prefixes = [
        (txt + label).strip()
        for txt, label in zip(sel_train["text"], sel_train["label_str"])
    ]
    prefix_exs = "\n\n".join(serialized_prefixes) + "\n"
    #print("confirming prefix_exs in target_random_prompt", prefix_exs)
    ##todo: the num_examples + 1 now is just a rough estimation for cost, token-wise and other cost func is required
    return len(sample_index) + 1, prefix_exs

def stratified_sampling_prompt(index, pd_data_files, data_size, record_matrix) ->str:##a greedy approach
    """Get stratified samples for prompt from train data."""
    prefix_exs_rows = sample_train_data_with_index(pd_data_files, data_size, index)
    serialized_prefixes = [
        (txt + label).strip()
        for txt, label in zip(prefix_exs_rows["text"], prefix_exs_rows["label_str"])
    ]
    prefix_exs = "\n\n".join(serialized_prefixes) + "\n"
    return prefix_exs

def testcurrent(Shapley_Value, samples):
    shapley_value_data = Shapley_Value
    shapley_value = [0 for i in range(len(shapley_value_data))]
    for i in range(len(shapley_value)):
        if(shapley_value_data[i][1] > 0):
            shapley_value[i] = shapley_value_data[i][0] / shapley_value_data[i][1]
        else:
            shapley_value[i] = -110
    topk_index = heapq.nlargest(k + 1, range(len(shapley_value)), shapley_value.__getitem__)
    delta_budget, prefix = target_random_prompt(
        samples, topk_index
    )
    score = call_api_once(prefix, test=True, test_num=100)
    return score

def sample_train_data_with_index(train: pd.DataFrame, n_rows: int, index):
    """
    Sample train data.

    Used when random sampling points for prompt.
    """
    train1 = train[0 : index]
    train2 = train[index + 1: len(train)]
    newtrain = pd.concat([train1, train2])
    res = newtrain.sample(n_rows - 1)
    res = pd.concat([train[index: index + 1], res])
    return res

def main():##todo: fill in the following 2 algorithms
    current_budget = 0
    Total_budget = 180  ##Maximum number of examples in QA
    Num_of_Examples = 200
    budget_threshold = 0
    budget_step = 10
    csv_list = []
    result_list = []
    '''
    pd_data_files = data_utils.read_data(
        data_dir="D:\\GitHub\\LLM-EM0001\\fm_data_tasks-main\\fm_data_tasks\\data\\datasets\\entity_matching\\structured\\Walmart-Amazon",
        class_balanced=True,
        add_instruction=False,
        max_train_samples= 1,
        max_train_percent=-1,
        sep_tok=".",
        nan_tok="nan",
    )'''
    #length = len(pd_data_files["train"])
    samples = prompt_utils.get_validation_dataset(
                        "outputs/Walmart-Amazon/validation/default/7k_0inst_0cb_random_200run_0dry/trial_0.feather",
                        num_examples=20,
                        task="entity_matching",
                    )#before running this, should make sure the directory is not empty
    length = len(samples)
    #print(length)
    Shapley_Value = [[0 for i in range(2)] for j in range(length)]
    SSB_sample_recorder = [[0 for i in range(k)] for j in range(length)]
    sample_index = []
    index = random.randint(1, length-1)
    starttime = time.time()
    iterations = 0
    ip_list = []
    while index > -1 and current_budget < Total_budget:
        size = random.randint(1, k) #size of this current sample
        all_index_list = [i for i in range(length)] #get all index
        all_index_list.remove(index)
        sample_index = random.sample(all_index_list, size - 1) #get other indices of the current target data point
        old_sample_index = sample_index
        sample_index.append(index) #get all indices
        #print(sample_index)
        SSB_sample_recorder[index][size - 1] += 1
        delta_budget,  prefix = target_random_prompt(
            samples, old_sample_index
        ) #compute budget and prompt
        current_budget += delta_budget #add budget
        delta_budget, prefix2 = target_random_prompt(
            samples, sample_index
        )  # compute budget and prompt
        current_budget += delta_budget  # add budget
        #print('currentbudget', current_budget)
        Shapley_Value, score = call_api(prefix, prefix2, Shapley_Value, index, size) #
        ip = initialize_percentage(SSB_sample_recorder)
        if(iterations % 4 == 0):
            endtime = time.time()
            print(index, current_budget, Shapley_Value, SSB_sample_recorder, ip)
            df1 = pd.DataFrame([index, current_budget, ip, score, endtime - starttime])
            df2 = pd.DataFrame(Shapley_Value)
            df3 = pd.DataFrame(SSB_sample_recorder)
            excel_file = 'k=' + str(k) + 'output_excel_file_iter='+str(iterations)+'.xlsx'
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # 将第一个数据框写入 Sheet1
                df1.to_excel(writer, sheet_name='Sheet1', index=False)
                # 将第二个数据框写入 Sheet2
                df2.to_excel(writer, sheet_name='Sheet2', index=False)
                df3.to_excel(writer, sheet_name='Sheet3', index=False)
        index = exist_unsampled_data(SSB_sample_recorder) #get unsampled index for next round of initialization
        iterations += 1
        if(current_budget > budget_threshold):
            budget_threshold += budget_step
            score = testcurrent(Shapley_Value, samples)
            endtime = time.time()
            result_list.append(['init', ip, current_budget, score, endtime-starttime, Shapley_Value])
    while current_budget < Total_budget:
        #index = largest_assertion_probability(SSB_sample_recorder, Shapley_Value)
        index = Border_Uncertainty_Sampling(Shapley_Value, k)
        max_partial_budget = max(SSB_sample_recorder[index])#finding the strata with largest
        for size in range(1, k+1):
            iterations = max_partial_budget - SSB_sample_recorder[index][size - 1]
            for j in range(iterations):
                all_index_list = [i for i in range(len(samples))]
                all_index_list.remove(index)
                old_sample_index = random.sample(all_index_list, size - 1)
                sample_index = old_sample_index
                sample_index.append(index)
                SSB_sample_recorder[index][size - 1] += 1
                delta_budget, prefix = target_random_prompt(
                    samples, old_sample_index
                )
                current_budget += delta_budget
                delta_budget, prefix2 = target_random_prompt(
                    samples, sample_index
                )
                current_budget += delta_budget
                Shapley_Value, score = call_api(prefix, prefix2, Shapley_Value, index, size)
            if (current_budget > budget_threshold):
                budget_threshold += budget_step
                score = testcurrent(Shapley_Value, samples)
                endtime = time.time()
                result_list.append(['process', current_budget, score, endtime - starttime, Shapley_Value])
    #print(result_list)
    with open('output-0112.txt', 'w') as file:
        # 将print语句的输出写入文件
        print(result_list, file=file)
    return 0

##todo: finish this part
def largest_assertion_probability():
    return 0

##todo: comparison algorithm
def Border_Uncertainty_Sampling(shapley_value_data, k):
    shapley_value = [0 for i in range(len(shapley_value_data))]
    for i in range(len(shapley_value_data)):
        shapley_value[i] = shapley_value_data[i][0]/shapley_value_data[i][1]
    topk_index = heapq.nlargest(k + 1, range(len(shapley_value)), shapley_value.__getitem__)
    topk = heapq.nlargest(k + 1, shapley_value)
    interval = (topk[len(topk)-2] + topk[len(topk)-1])/2
    max_evidence = 0
    max_index = 0
    for index in topk_index:
        evidence = abs(shapley_value[index]-interval)*shapley_value_data[index][1]
        if(evidence > max_evidence):
            max_evidence = evidence
            max_index = index
    return max_index

def parse_args() -> argparse.Namespace:##setting default parameters for parser
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
        default="D:/GitHub/LLM-EM0001/fm_data_tasks-main/fm_data_tasks/data/datasets/entity_matching/structured/Walmart-Amazon",
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
        default="openai",
        choices=["openai", "opt", "huggingface"],
    )
    parser.add_argument(
        "--client_connection",
        type=str,
        help="Manifest client connection string.",
        #chatgpt
        default="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",
        # huggingface(to be confirmed)
        #default="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",
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
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=k)
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
        default=1,
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


if __name__ == "__main__":
    main()
