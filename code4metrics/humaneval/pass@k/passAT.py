import subprocess, shutil
import argparse
import os
import re, time
import json
from tqdm import tqdm, trange
from typing import Optional, Tuple
from LogRecorder import CLogRecoder
import numpy as np

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
pro_path = '/data4/liupei/NDSS2026/Eval/humaneval/'
data_file = os.path.join(pro_path, 'decompile-eval-executable-gcc-obj.json')
_DEFAULT_CMD_TIMEOUT = 10
_MODEL = 'sf22538'
_INDEX = '0'
_ISNEW = True

os.environ['TMPDIR'] = '/data4/liupei/NDSS2026/Eval/humaneval/tmp/tmp'

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, default=pro_path, required=False)
parser.add_argument('--input', type=str, default='', required=False)
parser.add_argument('--data_file', type=str, default='decompile-eval-executable-gcc-obj.json', required=False)
parser.add_argument('--model', type=str, default=_MODEL, required=False)
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument("--origin", action="store_true", help="no post-handle")

args = parser.parse_args()
if args.workdir != pro_path:
    pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if args.data_file != 'decompile-eval-executable-gcc-obj.json':
    data_file = os.path.join(pro_path, args.data_file)
if not os.path.exists(data_file):
    print(f'Error: data {data_file} not exists!')
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
if args.origin:
    _ISNEW = False

analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'humaneval_{_MODEL}_{_INDEX}')
input_path = os.path.join(res_path, f'decom_{_BITWIDE}')
store_path = os.path.join(res_path, f'c_file_{_BITWIDE}')
tmp_path = os.path.join(pro_path, 'tmp')
if not os.path.exists(res_path):
    print(f'Error: result dir {res_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: decom dir {input_path} not exists!')
    quit(-1)
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if os.path.exists(store_path):
    shutil.rmtree(store_path)
os.makedirs(store_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logger = CLogRecoder(
    logfile=os.path.join(analysis_path, '{}_hel-metric_{}_{}_{}.log'.format(ymd, _MODEL, _BITWIDE, _INDEX)))


# execute shell command
def _run_command(command: str, timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[str, str]:
    # output = subprocess.run(command.split(), shell=True, capture_output=True, text=True, input=stdin, timeout=timeout)
    output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    stdout = output.stdout.decode('utf-8') if isinstance(output.stdout, bytes) else output.stdout
    stderr = output.stderr.decode('utf-8') if isinstance(output.stderr, bytes) else output.stderr
    return stdout, stderr


def handle_decom(funcname, content, source):
    matches = re.findall(r'```.*?\n((.|\n)*?)```', content, re.MULTILINE)
    if matches:
        content = matches[0][0].strip()
    else:
        logger.INFO("not match decom {}".format(funcname))
        content = content.strip()
    lines = content.split('\n')
    for index in range(len(lines)):
        if lines[index]:
            if '#include' in lines[index] and lines[index] in source:
                lines[index] = ''
                continue
            mat = re.findall(r'([a-zA-Z0-9_]{2,})[ ]*\(', lines[index], re.MULTILINE)
            if mat:
                changed = ' ' + funcname + '('
                line = re.sub(r'([a-zA-Z0-9_]{2,})[ ]*\(', changed, lines[index], re.MULTILINE)
                if line != lines[index]:
                    lines[index] = line
                    logger.INFO("changed decom {}".format(funcname))
                break
    if index == len(lines) - 1:
        logger.INFO("No change decom {}".format(funcname))
    return '\n'.join(lines)


def evaluate_func(c_func, c_test, c_func_decompile, input_asm_prompt_id):
    if _ISNEW:
        c_func_decompile = handle_decom('func0', c_func_decompile, c_func)

    flag_compile = 0
    flag_run = 0
    c_include = ''
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
            c_func = c_func.replace(line, '')
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
            c_test = c_test.replace(line, '')
    c_combine = c_include + '\n' + c_func_decompile + '\n' + c_test
    c_onlyfunc = c_include + '\n' + c_func_decompile

    # Define the C file and executable names
    c_file_onlyfunc = os.path.join(tmp_path, 'onlyfunc.c')
    executable_onlyfunc = os.path.join(tmp_path, 'onlyfunc')
    c_file = os.path.join(tmp_path, 'combine.c')
    c_file_save = os.path.join(store_path, f'func{input_asm_prompt_id}.c')
    executable = os.path.join(tmp_path, 'combine')
    if os.path.exists(executable_onlyfunc):
        os.remove(executable_onlyfunc)
    if os.path.exists(executable):
        os.remove(executable)

    with open(c_file_onlyfunc, 'w') as f:
        f.write(c_onlyfunc)
    with open(c_file, 'w') as f:
        f.write(c_combine)
    with open(c_file_save, 'w') as f:
        f.write(c_combine)

    # Compile the C program to an assembly
    if not _ISNEW:
        compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
        try:
            subprocess.run(compile_command, shell=True, check=True)
            flag_compile = 1
        except:
            return flag_compile, flag_run
    else:
        compile_command = f'gcc -S {c_file_onlyfunc} -o {executable_onlyfunc} -lm'
        try:
            stdout, stderr = _run_command(compile_command)
        except Exception as e:
            logger.INFO("Cmd Error:\n{}".format(e))
            return flag_compile, flag_run
        if not os.path.exists(executable_onlyfunc):
            logger.INFO('Error: no executable_onlyfunc')
            if stderr:
                logger.INFO("Compile Error:\n{}".format(stderr))
            return flag_compile, flag_run
        flag_compile = 1

    # Compile the C program to an executable
    if not _ISNEW:
        compile_command = f'gcc {c_file} -o {executable} -lm'
        try:
            subprocess.run(compile_command, shell=True, check=True)
            flag_compile = 1
        except:
            return flag_compile, flag_run
    else:
        compile_command = f'gcc {c_file} -o {executable} -lm'
        try:
            stdout, stderr = _run_command(compile_command)
        except Exception as e:
            logger.INFO("Cmd Error:\n{}".format(e))
            return flag_compile, flag_run
        if not os.path.exists(executable):
            logger.INFO('Error: no executable')
            if stderr:
                logger.INFO("Execute Error:\n{}".format(stderr))
            return flag_compile, flag_run
        flag_compile = 1

    # Run the compiled executable
    run_command = f'{executable}'
    try:
        process = subprocess.run(run_command, shell=True, check=True, capture_output=True, timeout=_DEFAULT_CMD_TIMEOUT)
        flag_run = 1
    except subprocess.CalledProcessError as e:
        pass
    except Exception as e:
        pass
    return flag_compile, flag_run
    # try:
    #     stdout, stderr = _run_command(run_command)
    # except subprocess.CalledProcessError as e:
    #     pass
    # except Exception as e:
    #     logger.INFO("Cmd Error:\n{}".format(e))
    #     return flag_compile, flag_run
    # flag_run = 1
    # if stderr:
    #     logger.INFO("Run Error:\n{}".format(stderr))
    # return flag_compile, flag_run

def pass_at_k(n, c, k):
    # n : total number of samples
    # c : number of correct samples
    # k : k in pass@k
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


OPT = ["O0", "O1", "O2", "O3"]  # Optimization states
with open(data_file, 'r') as f:
    data_all = json.load(f)
NUM = int(len(data_all) / 4)
num_compile = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
num_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
all_pass_at_1  = {"O0": [], "O1": [], "O2": [], "O3": []}
all_pass_at_10  = {"O0": [], "O1": [], "O2": [], "O3": []}

result_dict = {}
for f in os.listdir(input_path):
    file_path = os.path.join(input_path, f)
    number = ''.join(f[4:].replace('.jsonl',''))
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get("mode") == _BITWIDE:
                opts_value = json_obj.get("opts")
                key = f"{number}_{opts_value}"
                val = json_obj.get("c_func_decompile")
                # val = json_obj.get("prediction_dec")
                if not _ISNEW:
                    val = val.replace("```c", "").replace("```", "")
                result_dict[key] = val

# c_func_decompiled_results = []
compile_func = []
run_func = []


for idx in trange(len(data_all)):
# for idx in trange(int(len(data_all)/4)):
    task_id = data_all[idx]['task_id']
    c_func = data_all[idx]['c_func']
    c_test = data_all[idx]['c_test']
    # input_asm_prompt = data_all[idx]['input_asm_prompt']
    opt_state = data_all[idx]['type']
    c = 0
    for n in range(20):
        input_asm_prompt_id = str(task_id) + '_'+str(n)+'_' + opt_state
        c_func_decompile = result_dict[input_asm_prompt_id]
        logger.INFO("handle func {}".format(input_asm_prompt_id))
        flag_compile, flag_run = evaluate_func(c_func, c_test, c_func_decompile, input_asm_prompt_id)
        if flag_compile == 1:
            compile_func.append(input_asm_prompt_id)
        if flag_run == 1:
            run_func.append(input_asm_prompt_id)
            c += 1
        num_compile[opt_state] += flag_compile
        num_run[opt_state] += flag_run
    all_pass_at_1[opt_state].append(pass_at_k(20, c, 1))
    all_pass_at_10[opt_state].append(pass_at_k(20, c, 10))

pass_at_1 = {"O0": 0.0, "O1": 0.0, "O2": 0.0, "O3": 0.0}
pass_at_10 = {"O0": 0.0, "O1": 0.0, "O2": 0.0, "O3": 0.0}

for optstate in ['O0','O1','O2','O3']:
    pass_at_1[optstate] = np.mean(all_pass_at_1[optstate]) 
    pass_at_10[optstate] = np.mean(all_pass_at_10[optstate]) 
    
print(pass_at_1)
print(pass_at_10)
# with open('c_func_decompiled_results.json', 'w') as json_file:
#     json.dump(c_func_decompiled_results, json_file)
metric_file = os.path.join(res_path, f'metric.txt')
with open(metric_file, 'a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    f.write('{} {} {} {}\n'.format(new_time, _MODEL, _BITWIDE, ('new' if _ISNEW else 'old')))
    for opt_state in num_compile.keys():
        f.write('opt:{}  pass@1 : {:.4f} pass@10 : {:.4f}\n '.format(opt_state, 
                                                                      pass_at_1[opt_state], pass_at_10[opt_state]))
    f.write('\n')
compile_func_file = os.path.join(res_path, f'compile_{_BITWIDE}.txt')
with open(compile_func_file, 'w') as f:
    f.write(str(compile_func))
run_func_file = os.path.join(res_path, f'reexecu_{_BITWIDE}.txt')
with open(run_func_file, 'w') as f:
    f.write(str(run_func))
