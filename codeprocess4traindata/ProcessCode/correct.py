#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare dataset for continual pre-training
"""

import argparse
import json
import math
import os
import time
# 该脚本纯粹是为了补全数据
import os
import json
import sys

def is_empty_dict(d):
    return isinstance(d, dict) and not d

def wrap_string_values(obj):
    if is_empty_dict(obj):
        return obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = [v]
            elif isinstance(v, list):
                continue
            else:
                print(f"------------------------error---------------------------")
                sys.exit()
        print(obj)
        return obj
    else:
        print(f'$$$$$$$$$$$$$$$$$$$$ error $$$$$$$$$$$$$$$$$$$$$$$$')
        sys.exit()


def process_json_obj(obj):
    """
    对 "data" 中的 param, .bss, .data, .rodata 部分进行递归 string→list 转换
    """
    if "data" in obj:
        obj["data"] = json.dumps(obj["data"], ensure_ascii=False)
    if "assemgraph_com" in obj:
        obj["assemgraph_com"] = json.dumps(obj["assemgraph_com"], ensure_ascii=False)
    if "assemgraph" in obj:
        obj["assemgraph"] = json.dumps(obj["assemgraph"], ensure_ascii=False)
    if "asm_obj" not in obj:
        obj["asm_obj"] = ""
    return obj


def process_jsonl_folder(in_folder_path, out_folder_path):
    folder_name = os.path.basename(os.path.normpath(out_folder_path))

    for filename in os.listdir(in_folder_path):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(in_folder_path, filename)
            output_path = os.path.join(out_folder_path, f"{folder_name}_{filename}")

            with open(input_path, "r", encoding="utf-8") as infile, \
                    open(output_path, "w", encoding="utf-8") as outfile:

                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        processed_obj = process_json_obj(obj)
                        json.dump(processed_obj, outfile, ensure_ascii=False)
                        outfile.write("\n")
                    except json.JSONDecodeError as e:
                        print(f"跳过无法解析的行 in {filename}: {e}")

    print(f"所有文件已处理完毕")

if __name__ == "__main__":
    in_path = "/data3/liupei/NDSS2026/TrainData/1FromSunJian/train_synth_simple_io/input"
    out_path = "/data3/liupei/NDSS2026/TrainData/2CorrectedDataset/train_synth_simple_io"
    process_jsonl_folder(in_path, out_path)


