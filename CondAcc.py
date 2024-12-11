# -*- coding: utf-8 -*-
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
from openai import OpenAI
import openai
import utils.data_utils as data_utils
import utils.prompt_utils as prompt_utils
from utils import constants
from utils.utils import compute_metrics, setup_logger
from unidecode import unidecode
import threading
import tiktoken

import math

##todo: global parameters

llm_model = "gpt-4"
dataset_index = 3
dataset_type_list = ['entity_matching/structured/','entity_matching/structured/','entity_matching/structured/','entity_matching/structured/','entity_matching/structured/','entity_matching/structured/','entity_matching/structured/', 'schema_matching/', 'error_detection/', 'data_imputation/', 'entity_matching/structured/', 'error_detection/', 'data_imputation/']
task_list = ['entity_matching', 'entity_matching', 'entity_matching', 'entity_matching', 'entity_matching', 'entity_matching', 'entity_matching', 'schema_matching', 'error_detection', 'data_imputation', 'entity_matching', 'error_detection', 'data_imputation']
dataset_name_list = ['Amazon-Google','DBLP-ACM','DBLP-GoogleScholar','Fodors-Zagats','iTunes-Amazon','Walmart-Amazon','Beer','Synthea', 'Hospital', 'Buy', 'Adult', 'Restaurant']
vali_name_list = ['5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry', '5k_0inst_0cb_random_200run_0dry']
sampletype_list = ["manual", "validation_clusters", "zero"]
dataset_type = dataset_type_list[dataset_index]
dataset_name = dataset_name_list[dataset_index]
task = task_list[dataset_index]
vali_name = vali_name_list[dataset_index]
tokennizer = tiktoken.encoding_for_model(llm_model)

'''
dataset_type = 'data_imputation/'
dataset_name = 'Buy'
task = 'data_imputation'
'''
Total_m = 100 ##Maximum number of examples in QA
k = 5
the_api_key = 'xxx'


def list_gcd(lst):
    gcd_result = lst[0]
    for num in lst[1:]:
        gcd_result = math.gcd(gcd_result, num)
    return gcd_result


client = OpenAI(
    # This is the default and can be omitted
    api_key='xx',
)
openai.api_key = 'xxx'


def call_gpt(prompt, max_tokens):
    print(prompt)
    pred = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens = max_tokens
    ).choices[0].message.content
    print("response from chatgpt", pred)
    return pred

logger = logging.getLogger(__name__)

result = []

def call_api(prefix_exs, prefix_exs2, shapley_value, target_index, size):
    v1 = call_api_once(prefix_exs)
    v2 = call_api_once(prefix_exs2)
    if(v1 == -100 or v2 == -100):
        return shapley_value, -100
    shapley_value[target_index][0] += v2 - v1
    shapley_value[target_index][1] += 1
    return shapley_value, v2 - v1

def call_api_once(prefix_exs, test = False, test_num = 100000):
    """Generate args."""
    prefix_exs = unidecode(prefix_exs)
    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("num_trials must be greater than 0.")
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))
    # Will set seed for pandas
    numpy.random.seed(args.seed)
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
    #print(test_file, num_run)
    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")
    logger.info(f"Running {num_run} examples for {args.num_trials} trials.")
    preds = []
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
            serialized_r = unidecode(serialized_r)
            queries.append((prefix_exs + "\n" + serialized_r).strip())
        gt = test_data["label_str"]
        idx = 0
        thenum = test_data.shape[0] if test else min(min(num_run, args.num_print), len(queries))
        for _ in range(thenum):
            #logger.info(prompt(queries[idx]))
            pred = ""
            if not args.dry_run:
                seconds = 0
                while(seconds < 40):
                    try:
                        pred = call_gpt(prompt(queries[idx]), args.max_tokens)
                        break
                    except Exception as e:
                        print(e)
                        seconds += 20
                        time.sleep(seconds)
                        print(seconds)
                if(seconds == 40):
                    print("not responding after" + str(seconds) + "seconds")
                    return -100
                else:
                    print("reponded after" + str(seconds) + "seconds")
            preds.append(pred)
            result.append(pred)
            print(result)
            logger.info(f"====> {pred} <====")
            #time.sleep(2)
            idx += 1
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
    if(dataset_type.startswith('data_imputation')):
        if(test):
            return trial_metrics["acc"], len(preds)
        else:
            return trial_metrics["acc"]
    else:
        if(test):
            return trial_metrics["f1"], len(preds)
        else:
            return trial_metrics["f1"]

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
    topk_index = heapq.nlargest(k + 1, range(len(Shapley_Value)), Shapley_Value.__getitem__)
    delta_budget, prefix = target_random_prompt(
        samples, topk_index
    )
    score = call_api_once(prefix, test=True, test_num=1000000)
    vali_score = call_api_once(prefix)
    return score, vali_score

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

def generate_subsets(nums, k):
    n = len(nums)
    subsets = []
    for i in range(1 << n):
        if bin(i).count('1') <= k:
            subset = [nums[j] for j in range(n) if (i & (1 << j)) != 0]
            subsets.append(subset)
    return subsets

def size_k_subsets(nums, k, tmp_k):
    n = len(nums)
    subsets = []
    for i in range(1 << n):
        if bin(i).count('1') <= k:
            subset = [nums[j] for j in range(n) if (i & (1 << j)) != 0]
            if(len(subset == tmp_k)):
                subsets.append(subset)
    return subsets

def sample_subset(nums, k):
    subsets = generate_subsets(nums, k)
    return random.choice(subsets)

def AvgDiff_sample_subset(nums, k):
    subsets = generate_subsets(nums, k)
    return random.choice(subsets)

def get_AvgDiff_klist(Total_m, min_sample_allo):
    klist = []
    for i in range(len(min_sample_allo)):
        for j in range(min_sample_allo[i]):
            klist.append(i)
            if(len(klist)==Total_m):
                return klist

def mysum(inputlist):
    sumoflist = 0
    for num in inputlist:
        sumoflist = sumoflist + num
    return sumoflist

def CondAcc():
    Total_m = 150
    result_list = []
    AvgDiff_klist = []
    tmp_idx = 0
    num_of_token = 0
    totaltime = time.time() - time.time()
    samples = prompt_utils.get_validation_dataset(
        "outputs/"+dataset_name+"/validation/default/" + vali_name + "/trial_0.feather",##todo: change the validation dataset to an existing one
        num_examples=20,
        task=task,
    )  # This is n sample candidates from validation set(players), before running this, should make sure the directory is not empty
    n = len(samples)
    print(samples)
    samples.to_excel('samples'+dataset_name+'.xlsx', index=False)
    CondAcc_record = [[0, 0, 0, 0] for j in range(n)]
    CondAcc = [0 for i in range(n)]
    starttime = time.time()
    current_m = 0
    u1 = 0
    u2 = 0
    while (current_m < Total_m):
        rand =random.randint(1,19)
        permutation = random.sample([i for i in range(n)], rand)
        random.shuffle(permutation)
        i=rand-1
        delta_budget, prefix1 = target_random_prompt(
            samples, permutation[ : i+1]
        )
        delta_budget, prefix2 = target_random_prompt(
            samples, permutation[: i + 1]
        )
        u1 = call_api_once(prefix1)[0]
        u2 = call_api_once(prefix2)[0]
        num_of_token+= len(tokennizer.encode(prefix1)) + len(tokennizer.encode(prefix2))
        current_m = current_m + 1
        delta_u = u1 - u2
        u2 = u1
        target = permutation[i]
        CondAcc_record[target][0] = CondAcc_record[target][0] + u1
        CondAcc_record[target][1] = CondAcc_record[target][1] + 1
        CondAcc_record[target][2] = CondAcc_record[target][0] + u2
        CondAcc_record[target][3] = CondAcc_record[target][1] + 1
        if (current_m % 5 == 0):
            endtime = time.time()
            totaltime += endtime - starttime
            starttime = endtime
            for target in range(n):
                sum = 0
                if (CondAcc_record[target][1] > 0):
                    CondAcc[target] = CondAcc_record[target][0] / CondAcc_record[target][1] - CondAcc_record[target][2] / CondAcc_record[target][3]
            score, vali_score = testcurrent(CondAcc, samples)
            result_list.append([current_m, score, totaltime,CondAcc])
            df1 = pd.DataFrame(
                ['cur-budget=' + str(current_m), 'testcur' + str(score), 'valiscore' + str(vali_score),
                 'time(s)' + str(totaltime), 'tokennum' + str(num_of_token)])
            df2 = pd.DataFrame(CondAcc)
            df3 = pd.DataFrame([prefix1])
            excel_file = 'k=' + str(k) + dataset_name +'MCSV-SVnum=' + str(current_m) + '.xlsx'
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                df1.to_excel(writer, sheet_name='Sheet1', index=False)
                df2.to_excel(writer, sheet_name='Sheet2', index=False)
                df3.to_excel(writer, sheet_name='Sheet3', index=False)
    return 0


def parse_args() -> argparse.Namespace:##setting default parameters for parser
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        default="D:/GitHub/LLM-EM0001/fm_data_tasks-main/fm_data_tasks/data/datasets/" + dataset_type + dataset_name,
        ##todo: It is possible to change the default --data_dir above
        ##required=True,
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        help="Which data directory to run.",
        default="D:/GitHub/LLM-EM0001/fm_data_tasks-main/fm_data_tasks/data/datasets/" + dataset_type + dataset_name,
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

def main():
    CondAcc()


if __name__ == "__main__":
    main()
