from audioop import mul
from tqdm import tqdm
from collections import OrderedDict
import os
from transformers import RobertaTokenizer

import typer
import glob
import argparse
import lizard
import re
import random
import json
import wandb
def preprocess(line):
    line = re.sub("[ \t\r\f\v]+", " ", line)
    line = re.sub("[ \t\r\f\v]+\n", "\n", line)
    return line

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def main(
    repositry: str = typer.Argument(
        ..., help="path to a repogitory or a parent directory with --multi option."
    ),
    output_dir: str = typer.Argument(..., help="path to create dataset."),
    multi: bool = typer.Option(
        False, help="if you want to convert multiple repositories, use --multi option."
    ),
    limit_methods: int = typer.Option(
        -1, help="limit the number of methods for dataset"
    ),
):
    output = output_dir if output_dir.endswith("/") else output_dir + "/"
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

    repo_list = None
    if multi:
        if not repositry.endswith("/"):
            repositry += "/"
        repo_list = [repositry + repo for repo in os.listdir(path=repositry)]
    else:
        repo_list = [repositry]
    for repo in repo_list:
        info = {}      
        print("start conveting " + repo)

        java_list = glob.glob(repo + "/**/*.java", recursive=True)

        NG_WORD = ["/test/"]

        def ng_word_detect(path):
            for word in NG_WORD:
                if path in word:
                    return False
            return True

        java_list = list(filter(ng_word_detect, java_list))
        java_list = random.sample(java_list, int(len(java_list)/10))
        java_list = tqdm(java_list, desc="extract methods", total=len(java_list))

        method_contexts = []

        for java_file in java_list:
            java_list.set_postfix(OrderedDict(file=java_file))

            methods = lizard.analyze_file(java_file).__dict__["function_list"]
            try:
                with open(java_file) as f:
                    lines = f.readlines()
                    for method in methods:
                        d = method.__dict__
                        line_count = d["end_line"] - d["start_line"]
                        name = d["name"]

                        if line_count <= 3:
                            continue
                        if "Test" in name:
                            continue
                        if "test" in name:
                            continue
                        if "toString" == name:
                            continue
                        context = lines[d["start_line"] - 1 : d["end_line"]]
                        d["context"] = context

                        def preprocess2(line):
                            return re.sub("\s+", " ", line)

                        context = list(map(preprocess2, context))
                        context[0] = re.sub("^\s+", "", context[0])

                        method_contexts.append(tuple(context))
            except:
                import traceback

                traceback.print_exc()

        org_count = len(method_contexts)
        method_contexts = list(set(method_contexts))
        filtered_methods = []
        for method in method_contexts:
            method_ = "\n".join(method)
            method_ = preprocess(comment_remover(method_))
            method_lines = method_.split("\n")
            method_lines_tokenized = [tokenizer.encode(line)[1:-1] for line in method_lines]
            total_len = 0
            for m in method_lines_tokenized:
                total_len+=len(m)
            if(total_len > 100):
                continue
            filtered_methods +=[method]
        method_contexts = filtered_methods
        dlt_count = org_count - len(method_contexts)
        print(str(dlt_count) + " methods were skipped")
        


        random.shuffle(method_contexts)
        if limit_methods > 0:
            method_contexts = random.sample(method_contexts, limit_methods)
    
        dataset_size = len(method_contexts)
        
        testing_size = dataset_size // 10
        evalation_size = dataset_size // 10
        training_size = dataset_size - testing_size - evalation_size
        
        training_methods = method_contexts[:training_size]
        evalation_methods = method_contexts[
            training_size : training_size + evalation_size
        ]
        testing_methods = method_contexts[training_size + evalation_size :]

        
        info_lines={}
        for methods, mode in zip(
            [training_methods, evalation_methods, testing_methods],
            ["training", "evalation", "testing"],
        ):
            count = 0
            if not os.path.exists(output + repo):
                os.makedirs(output + repo)
            with open(output + repo + "/" + mode + "_masked_code.txt", "w") as fd:
                with open(output + repo + "/" + mode + "_mask.txt", "w") as fm:
                    for method in methods:
                        method_ = "\n".join(method)
                        
                        method_ = comment_remover(method_)
                        method_ = preprocess(method_)
                        
                        method_lines = method_.split("\n")
                        method_lines_tokenized = [tokenizer.encode(line)[1:-1] for line in method_lines]
                        
                        total_len = 0
                        for m in method_lines_tokenized:
                            total_len+=len(m)
                        
                        method_lines = [tokenizer.decode(tokens) for tokens in method_lines_tokenized]
                        
                        for i, tokenized in enumerate(method_lines_tokenized):
                            if  not 2 < len(tokenized):
                                continue

                            rand_max = min(10, len(tokenized))
                            rand_min = 1
                            mask_num = random.randint(rand_min, rand_max)
                            last_index = len(tokenized) - mask_num

                            mask = tokenizer.decode(tokenized[last_index:])
                            
                           
                            masked_line = ""
                            masked_tokens = tokenized[: last_index + 1]
                            if len(masked_tokens) > 0:
                                masked_line = tokenizer.decode(tokenized[: last_index]) + "<x>"
                            else:
                                masked_line = "<x>"
                            masked_list = (
                                method_lines[:i] + [masked_line] + method_lines[i + 1 :]
                            )
                            
                            masked_instance = "".join(masked_list)
                            masked_instance = preprocess(masked_instance)

                            fd.write(masked_instance + "\n")
                            fm.write(mask + "<z>\n")
                            count+=1
            info_lines[mode]=count
        info[repo] ={
            "dataset_methods":dataset_size,
            "train_methods":training_size,
            "eval_methods":evalation_size,
            "test_methods":testing_size,
            "train_lines":info_lines["training"],
            "eval_lines":info_lines["evalation"],
            "test_lines":info_lines["testing"]
        }        
        path3 = output + repo + "/info.json"
        json_file3 = open(path3, mode="w")
        json.dump(info, json_file3, indent=2, ensure_ascii=False)
        json_file3.close()

if __name__ == "__main__":
    wandb.login()
    run =wandb.init()
    typer.run(main)

