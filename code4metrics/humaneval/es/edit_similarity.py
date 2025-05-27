import subprocess, shutil
import argparse
import os
import re, time
import json
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from LogRecorder import CLogRecoder

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
pro_path = '/data4/liupei/NDSS2026/Eval/humaneval'
input_path = ''
decom_flag = 'c_func_decompile'
_MODEL = 'gpt-4o'
_INDEX = '0'
_THREAD_NUM = 30
_IDATT = False
_CFGTT = False
_DATATT = False

os.environ['TMPDIR'] = '/data4/liupei/NDSS2026/Eval/humaneval/tmp/tmp'

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, default=pro_path, required=False)
parser.add_argument('--input', type=str, default='', required=False)
parser.add_argument('--model', type=str, default=_MODEL, required=False)
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument("--ida_test", action="store_true", help="test ida decom")
parser.add_argument("--cfg_test", action="store_true", help="test block num > 1")
parser.add_argument("--data_test", action="store_true", help="test block num > 1")
args = parser.parse_args()
if args.workdir != pro_path:
    pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if args.input:
    input_path = args.input
if args.model != _MODEL:
    _MODEL = args.model
if args.i:
    _INDEX = args.i
if args.b in ['64', '32']:
    _BITWIDE = args.b
else:
    print(f'Error bit-wide {args.b} !')
    quit(-1)
if args.ida_test:
    _IDATT = True
    decom_flag = 'pscode'
if args.cfg_test:
    _CFGTT = True
if args.data_test:
    _DATATT = True

analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'humaneval_{_MODEL}_{_INDEX}')
if not input_path:
    input_path = os.path.join(res_path, f'decom_{_BITWIDE}')
if not os.path.exists(res_path):
    print(f'Error: result dir {res_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: decom dir {input_path} not exists!')
    quit(-1)
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

def handle_decom(content):
    matches = re.findall(r'```.*?\n((.|\n)*?)```', content, re.MULTILINE)
    if matches:
        content = matches[0][0].strip()
    else:
        content = content.strip()
    if _IDATT:
        content = content.replace('__fastcall', '')
        content = content.replace('__cdecl', '') 
        content = content.replace('__usercall', '') 
    return content

def edit_distance(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)

    # dp[i][j] 表示 seq1[:i] 与 seq2[:j] 的编辑距离
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 初始化：当另一个序列为空时，需要的插/删操作数
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 逐步填充 DP 表
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                # 若当前 token 相同，则无需额外编辑
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 取删除、插入、替换的最小代价 + 1
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # 删除 seq1[i-1]
                    dp[i][j - 1],  # 在 seq1 插入 seq2[j-1]
                    dp[i - 1][j - 1]  # 替换 seq1[i-1] 为 seq2[j-1]
                )
    return dp[len1][len2]

def edit_similarity(seq_pred, seq_true):
    if not seq_true:  # 如果真实序列为空
        return 0.0
    distance = edit_distance(seq_pred, seq_true)
    similarity = 1.0 - float(distance) / len(seq_true)
    return max(0.0, similarity)

def process_func(idx):
    global _BITWIDE
    result = {}

    file_path = os.path.join(input_path, f'func{idx}.jsonl')
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get("mode") != _BITWIDE:
                continue
            if _CFGTT and json_obj['assemgraph']['nodenum'] == 1:
                continue
            json_obj['data'] = json.loads(json_obj['data'])
            if _DATATT and not json_obj['data']['.rodata'] and not json_obj['data']['.data']:
                continue
            decom = handle_decom(json_obj[decom_flag])
            result[json_obj['opts']] = edit_similarity(json_obj['sourcecode'], decom)
    return result

results = {"O0": [], "O1": [], "O2": [], "O3": []}
with ThreadPoolExecutor(max_workers=_THREAD_NUM) as executor:
    futures = [executor.submit(process_func, ind) for ind in range(164)]

    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing humaneval_{_MODEL}_{_INDEX} {_BITWIDE}'):
        result = future.result()
        for opt, val in result.items():
            results[opt].append(val)

avg_similarities = {
        state: sum(results[state]) / len(results[state]) if results[state] else 0.0
        for state in results
    }
metric_file = os.path.join(res_path, 'es_metric.txt')
with open(metric_file, 'a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if _CFGTT:
        f.write('{} {} {} cfg_test\n'.format(new_time, _MODEL, _BITWIDE))
    elif _DATATT:
        f.write('{} {} {} data_test\n'.format(new_time, _MODEL, _BITWIDE))
    else:
        f.write('{} {} {}\n'.format(new_time, _MODEL, _BITWIDE))
    for opt_state, val in avg_similarities.items():
        f.write('opt:{} run_rate:{:.4f}\n'.format(opt_state, val))
    f.write('\n')