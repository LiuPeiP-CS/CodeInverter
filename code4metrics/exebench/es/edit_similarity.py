import json, time, shutil, os, argparse, ast, subprocess, traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIStatusError
from string import Template
from LogRecorder import CLogRecoder

# 指定文件夹路径
pro_path = '/exebench'
data_path = os.path.join(pro_path, 'origin')
_MODEL = 'gpt-4o'
_INDEX = '0'
_THREAD_NUM = 30
# _INTERVAL = 100
# _EXELEN = 5000
_GPTHD = False
_CFGTT = False
_DATATT = False

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, default=pro_path, required=False)
parser.add_argument('--data_path', type=str, default='origin', required=False)
parser.add_argument('--model', type=str, default=_MODEL, required=False)
# parser.add_argument("-s", required=True, help="start")
# parser.add_argument("-x", required=True, help="interval")
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument("--gpt_handle", action="store_true", help="no post-handle")
parser.add_argument("--cfg_test", action="store_true", help="test block num > 1")
parser.add_argument("--data_test", action="store_true", help="test block num > 1")
args = parser.parse_args()
if args.workdir != pro_path:
    pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if args.data_path != 'origin':
    data_path = os.path.join(pro_path, args.data_path)
if not os.path.exists(data_path):
    print(f'Error: data {data_path} not exists!')
    quit(-1)
if args.model != _MODEL:
    _MODEL = args.model
if args.i:
    _INDEX = args.i
if args.b in ['64', '32']:
    _BITWIDE = args.b
else:
    print(f'Error bit-wide {args.b} !')
    quit(-1)
if args.gpt_handle:
    _GPTHD = True
if args.cfg_test:
    _CFGTT = True
if args.data_test:
    _DATATT = True
# if not args.s or not args.x:
#     start_index = 0
#     end_index = _EXELEN
#     module = ''
# else:
#     start_index = int(args.s) * _INTERVAL
#     end_index = start_index + int(args.x) * _INTERVAL
#     module = '_{}-{}'.format(args.s, int(args.s) + int(args.x))
#     if end_index > _EXELEN:
#         end_index = _EXELEN
# print("[+] {} to {} func".format(start_index, end_index))

analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'exebench_{_MODEL}_{_INDEX}')
input_path = os.path.join(res_path, 'c_file_{}{}'.format(_BITWIDE, ('_gpthd' if _GPTHD else '')))
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if not os.path.exists(res_path):
    print(f'Error: result dir {res_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: input {input_path} not exists!')
    quit(-1)
# if _GPTHD:
#     gpthd_path = os.path.join(res_path, 'gpthd_{}'.format(_BITWIDE))
#     if not os.path.exists(gpthd_path):
#         print(f'Error: gpthd dir {gpthd_path} not exists!')
#         quit(-1)


ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logger = CLogRecoder(logfile=os.path.join(analysis_path, '{}_exe-es_{}_{}_{}.log'.format(ymd, _MODEL, _BITWIDE, _INDEX)))

funcname_file = os.path.join(data_path, 'sourcecode.json')
with open(funcname_file, 'r') as f:
    source = json.load(f)
if _CFGTT:
    file = os.path.join(data_path, 'nocfg.json')
    with open(file, 'r') as f:
        cfg_test = json.load(f)
if _DATATT:
    file = os.path.join(data_path, 'nodata.json')
    with open(file, 'r') as f:
        data_test = json.load(f)

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

def process_func(func):
    global _BITWIDE
    source_code = source[func]
    result = {}
    
    for opt in ['O0', 'O1', 'O2', 'O3']:
        if _CFGTT and func not in cfg_test[_BITWIDE][opt]:
            continue
        if _DATATT and func not in data_test[_BITWIDE][opt]:
            continue
        file_path = os.path.join(input_path, f'{func}_{opt}.c')
        if not os.path.exists(file_path):
            continue
        decom = ''
        with open(file_path, 'r') as f:
            for line in f:
                if func in line:
                    decom += line
        if not decom:
            logger.INFO(f'No decom in {func}_{opt}.c')
            file_path = os.path.join(input_path, f'{func}_{opt}_only.c')
            if not os.path.exists(file_path):
                logger.INFO(f'No file {func}_{opt}_only.c')
                continue
            with open(file_path, 'r') as f:
                decom = f.read()
            if not decom:
                logger.INFO(f'No decom in {func}_{opt}_only.c')
                continue
        result[opt] = edit_similarity(source_code, decom)
    
    return result

results = {"O0": [], "O1": [], "O2": [], "O3": []}
with ThreadPoolExecutor(max_workers=_THREAD_NUM) as executor:
    futures = [executor.submit(process_func, name) for name in source.keys()]

    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing exebench_{_MODEL}_{_INDEX} {_BITWIDE}'):
        result = future.result()
        for opt, val in result.items():
            if val:
                results[opt].append(val)

avg_similarities = {
        state: sum(results[state]) / len(results[state]) if results[state] else 0.0
        for state in results
    }
metric_file = os.path.join(res_path, 'es_metric.txt')
with open(metric_file, 'a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if _CFGTT:
        f.write('{} {} {} {} cfg_test\n'.format(new_time, _MODEL, _BITWIDE, ('GPTHD' if _GPTHD else '')))
    elif _DATATT:
        f.write('{} {} {} {} data_test\n'.format(new_time, _MODEL, _BITWIDE, ('GPTHD' if _GPTHD else '')))
    else:
        f.write('{} {} {} {}\n'.format(new_time, _MODEL, _BITWIDE, ('GPTHD' if _GPTHD else '')))
    for opt_state, val in avg_similarities.items():
        f.write('opt:{} run_rate:{:.4f}\n'.format(opt_state, val))
    f.write('\n')
