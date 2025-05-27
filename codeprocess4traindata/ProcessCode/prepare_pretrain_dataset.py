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
from multiprocessing import cpu_count

from dataset.spliced_and_tokenized_dataset import (
    ClosedToConstantLengthSplicedDataset,
    supervised_tokenize_pretrain,
)
from datasets import dataset_dict, load_dataset
from transformers import AutoTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def main():
    # dir_path = "new_input_4"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input_dirs",
        type=str,
        # default=f"../../../Datasets/OrigDataset/ProcessedDataset/{dir_path}", # 此处需要替换我们实际需要的包含所有jsonl的文件夹的地址-----------------------------------------------
        default = "/data3/liupei/NDSS2026/TrainData/3SplitedDataset1", #
        help="Comma(i.e., ',') separated list of all data directories containing `.jsonl` data files.",
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, default="/data3/liupei/llm4decompile/llm4dec-1.3b", help="A directory containing the tokenizer"
    )
    parser.add_argument("--data_output_dirs", type=str, default=f"/data3/liupei/NDSS2026/TrainData/4PreparedData4Train", help="Data output directory")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length of each spliced tokenized sequence")
    parser.add_argument("--num_spliced_dataset_bins", type=int, default=4, help="Number of spliced dataset bins")
    args = parser.parse_args()

    if args.num_spliced_dataset_bins >= 100000:
        raise ValueError("Too many spliced divisions, must be smaller than 100000")

    args.data_cache_dir = os.path.join(args.data_output_dirs, "cache")
    args.data_jsonl_output_dir = os.path.join(args.data_output_dirs, "jsonl")
    args.data_arrow_output_dir = os.path.join(args.data_output_dirs, "arrow")

    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir)
    if not os.path.exists(args.data_jsonl_output_dir):
        os.makedirs(args.data_jsonl_output_dir)
    if not os.path.exists(args.data_arrow_output_dir):
        os.makedirs(args.data_arrow_output_dir)

    # Prepare to the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    data_count = 0

    # Prepare to all input datasets
    # input_data_paths = []
    # input_data_dirs = args.data_input_dirs.split(",")
    input_data_dirs = [os.path.join(args.data_input_dirs, d) for d in os.listdir(args.data_input_dirs) if
                       os.path.isdir(os.path.join(args.data_input_dirs, d))]
    for ds_dir in input_data_dirs:
        ds_dir = os.path.abspath(ds_dir)
        assert os.path.exists(ds_dir), f"Not find data dir {ds_dir}"
        ###################################################

        print(f"------------------- We are processing the directory {ds_dir} --------------------")
        current_dir_id = ds_dir.split('_')[-1]        
        input_data_paths = [] # 某一个文件夹里所有的文件地址的列表
        ds_files = [name for name in os.listdir(ds_dir) if name.endswith(".jsonl")]
        input_data_paths = [os.path.join(ds_dir, name) for name in ds_files]
        # input_data_paths.extend(ds_paths)
        ###################################################
        try:
            # Prepare to data splitting.
            train_splits = []
            split_interval = math.ceil(100 / args.num_spliced_dataset_bins)
            for i in range(0, 100, split_interval):
                start = i
                end = i + split_interval
                if end > 100:
                    end = 100
                train_splits.append(f"train[{start}%:{end}%]")

            # print("------------------- This is before list_dataset -------------------")
            list_dataset = load_dataset(
                path="json",
                data_files=input_data_paths,
                cache_dir=os.path.join(args.data_cache_dir, "raw"),
                keep_in_memory=False,
                split=train_splits,
                # num_proc=cpu_count(),
                # num_proc=64,
                num_proc=128
            )

            # print("------------------- This is after list_dataset -------------------")
            for index, dataset in enumerate(list_dataset):
                try:
                    assert isinstance(dataset, dataset_dict.Dataset)
                    logger.info(f"Start to process part-{index}/{len(list_dataset)} of all original datasets.")
                    # logger.info("*******************before error*******************")
                    dataset = dataset.map(
                        function=supervised_tokenize_pretrain,
                        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
                        keep_in_memory=False,
                        # num_proc=min(len(dataset), cpu_count()),
                        # num_proc=64,
                        num_proc=128
                    )
                    dataset = dataset.filter(lambda x: x is not None)
                    dataset = dataset.filter(
                        lambda x: x['seq_length'] > 1 and len(x['input_ids']) > 1
                    )
                    # logger.info("----------------after error----------------")
                    
                    # dataset = dataset.remove_columns(column_names=["source", "target", "category"])
                    dataset = dataset.remove_columns(column_names=["arch", "mode", "opts", "sourcecode", "asm_obj", "asm_ida_com", 'asm_ida', "pscode", 'data', 'assemgraph_com', 'assemgraph'])
                    dataset = dataset.sort(column_names=["seq_length"], reverse=False, keep_in_memory=False)
                    dataset = dataset.remove_columns(column_names=["seq_length"])
                    spliced_dataset = ClosedToConstantLengthSplicedDataset(
                        dataset=dataset, tokenizer=tokenizer, max_length=args.max_length, error_strict=False
                    )
                    # Save each jsonl spliced dataset.
                    output_index = "0" * (5 - len(str(index))) + str(index)
                    output_name = f"part-{current_dir_id}-{output_index}"
                    output_jsonl_path = os.path.join(args.data_jsonl_output_dir, output_name + ".jsonl")
                    st = time.time()
                    with open(file=output_jsonl_path, mode="w", encoding="utf-8") as fp_writer:
                        spliced_count = 0
                        for spliced_data_point in spliced_dataset:
                            if spliced_count % 500 == 0:
                                logger.info(f"processing {spliced_count} spliced data points for {fp_writer.name}")
                            spliced_count += 1
                            fp_writer.write(json.dumps(spliced_data_point, ensure_ascii=False) + "\n")
                    logger.info(
                        f"Current file {fp_writer.name}; "
                        f"Data size: {len(spliced_dataset)}; "
                        f"Spliced data size: {spliced_dataset.current_size}; "
                        f"Splicing compression rate: {round(spliced_dataset.current_size / len(spliced_dataset), 6)}; "
                        f"Time cost: {round((time.time() - st) / 60, 6)} minutes."
                    )

                    # Save each arrow spliced dataset
                    output_arrow_path = os.path.join(args.data_arrow_output_dir, output_name)
                    logger.info(f"Start to save {output_arrow_path}")
                    spliced_dataset = load_dataset(
                        path="json",
                        data_files=[output_jsonl_path],
                        cache_dir=os.path.join(args.data_cache_dir, "spliced_and_tokenized"),
                        keep_in_memory=False,
                        # num_proc=cpu_count(),
                        # num_proc=64,
                        num_proc=128,
                        split="train",
                    )
                    spliced_dataset.save_to_disk(dataset_path=output_arrow_path,
                                                 # num_proc=min(len(spliced_dataset), cpu_count())
                                                 # num_proc=64
                                                 num_proc=128
                                                 )
                    os.remove(output_jsonl_path)
                    data_count = data_count + len(dataset)
                    print(f"Have deleted the jsonl file, and get {data_count} data!")
                except Exception as e:
                    logger.info(f'the error is {e} in list_dataset')
                    continue

        except Exception as e:
            logger.info(f'******* the error in {ds_dir} is {e} *******')
            continue


if __name__ == "__main__":
    main()
