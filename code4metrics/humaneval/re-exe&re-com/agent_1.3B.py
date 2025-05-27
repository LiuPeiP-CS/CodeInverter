import json, time, shutil
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import re
from tqdm import trange
from string import Template
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["NCCL_P2P_DISABLE"] = '1'
#os.environ["NCCL_IB_DISABLE"] = '1'
# ָ���ļ���·��
pro_path = '/humaneval/'
input_path = os.path.join(pro_path, 'input')
prompt_file = ''
_BASE_MODEL_PATH = '***'
_MODEL_ID = 'epoch-0_step-12000-humaneval-0.4832/' # ����Ҫ���ĸ���Ϣ����epoch-0_step-2000
_MODEL_PATH = _BASE_MODEL_PATH + _MODEL_ID + '/modeling'
_INDEX = '0'
_ISIDA = True
_INTERVAL = 1
_MODEL = _MODEL_ID
# prompt4input = Template("""
# You are a professional decompilation assistant whose task is to understand the function's logic and data based on the assembly code, control flow graph (CFG) and data mapping table, and predict the corresponding high-level C source code.
#
# - The control flow graph:
# Each 'cfg_block' in 'cfg_blocks' is represented by a name and the corresponding assembly.
# Each 'egde' in 'edges' (not []) indicates connection between two cfg blocks through their block names.
#
# - The data mapping table defines the correspondence between data labels (in the assembly code and the CFG) and their actual values. It contains four types of data:
# -- 'param': stack variables (size and relative offset).
# -- '.rodata': read-only constants (size and value).
# -- '.data': initialized global/static data (size and initial value).
# -- '.bss': uninitialized global/static data (size and count, value initialized to 0).
#
# Please ensure that the predicted C source code is well-structured, syntactically correct, and reflects the accurate logic and data of the assembly code.
#
# - The assembly code:
# ```Json
# $assembly_prompt
# ```
#
# - The control flow graph:
# ```Json
# $cfg_prompt
# ```
#
# - The data mapping table:
# ```Json
# $data_mapping
# ```
#
# Please perform decompilation based on the above information, and only provide the decompiled C source code without any additional text:
# """)


parser = argparse.ArgumentParser()
parser.add_argument('--workdir', type=str, default=pro_path, required=False)
parser.add_argument('--input', type=str, default='input', required=False)
parser.add_argument('--prompt_file', type=str, default='', required=False)
parser.add_argument('--model', type=str, default=_MODEL, required=False)
parser.add_argument("-s", required=True, help="start")
parser.add_argument("-x", required=True, help="interval")
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument('--model_path',type=str,default=_MODEL_PATH,required=False)
# parser.add_argument("-f", action="store_true", help="first ?")
args = parser.parse_args()
if args.workdir != pro_path:
    pro_path = args.workdir
if not os.path.exists(pro_path):
    print(f'Error: workdir {pro_path} not exists!')
    quit(-1)
if args.input != 'input':
    input_path = os.path.join(pro_path, args.input)
if not os.path.exists(input_path):
    print(f'Error: input {input_path} not exists!')
    quit(-1)
# if args.prompt_file:
#     prompt_file = os.path.join(pro_path, args.prompt_file)
#     if not os.path.exists(prompt_file):
#         print(f'Error: prompt_file {prompt_file} not exists!')
#         quit(-1)
#     with open(prompt_file, 'r') as f:
#         prompt4input = Template(f.read())
#     if not prompt4input:
#         print(f'Error: No prompt in {prompt_file} !')
#         quit(-1)
# if args.model != _MODEL:
#     _MODEL = args.model
if args.model:
    _MODEL_ID = args.model
    _MODEL = _MODEL_ID
    _MODEL_PATH = _BASE_MODEL_PATH + _MODEL_ID + '/modeling'
    args.model_path = _MODEL_PATH

shutil.copy("***/special_tokens_map.json", _MODEL_PATH + '/')
shutil.copy("***/tokenizer_config.json", _MODEL_PATH + '/')
shutil.copy("***/tokenizer.json", _MODEL_PATH + '/')

if args.i:
    _INDEX = args.i
if args.b in ['64', '32']:
    _BITWIDE = args.b
else:
    print(f'Error bit-wide {args.b} !')
    quit(-1)
# is_first = True if args.f else False
if not args.s or not args.x:
    start_index = 0
    end_index = 164
    module = ''
else:
    start_index = int(args.s) * _INTERVAL
    end_index = start_index + int(args.x) * _INTERVAL
    module = '_{}-{}'.format(args.s, int(args.s) + int(args.x))
    if end_index > 164:
        end_index = 164
print("[+] {} to {} func".format(start_index, end_index))

ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
analysis_path = os.path.join(pro_path, 'analysis')
res_path = os.path.join(pro_path, f'humaneval_1.3B_{_INDEX}')
result_path = os.path.join(res_path, f'decom_{_BITWIDE}')
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if not os.path.exists(res_path):
    os.makedirs(res_path)
if start_index == 0:
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
elif not os.path.exists(result_path):
    print(f'Error: output {result_path} not exists!')
    quit(-1)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_instruct(datapoint):
    # datapoint["data"] = json.loads(datapoint["data"])
    # datapoint["assemgraph_com"] = json.loads(datapoint["assemgraph_com"])

    add_info = {
        "instruction set architecture": datapoint['arch'],
        "bit width": datapoint['mode'],
        "compiler optimization level": datapoint['opts']}
    add_info = json.dumps(add_info)

    data_mapping = {
        "stack variables (size and relative offset)": datapoint['data']['param'],
        "read-only constants (size and value)": datapoint['data']['.rodata'],
        "initialized global/static data (size and initial value)": datapoint['data']['.data'],
        "uninitialized global/static data (size and count)": datapoint['data']['.bss'],
    }
    data_mapping = json.dumps(data_mapping)

    if datapoint['assemgraph_com']['nodenum'] == 1:
        # ����ڵ�Ϊ1��prompt,��ʾ��ida��������û��cfg graph
        assembly_code = datapoint['asm_ida_com'] # ida�����ķ������룬ֻ��һ��cfg block��û��ͼ

        prompt4input = Template("""You are a professional decompilation assistant. Please understand the following assembly code and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code. Please ensure that the generated C source code is well-structured, syntactically correct, and reflects the accurate logic and data of the assembly code.

        - The assembly code:
        ```Assembly
        $assembly_code
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, assembly_code=assembly_code, data_mapping=data_mapping)


    elif datapoint['assemgraph_com']['nodenum'] > 1:

        cfg_prompt = {
            "cfg_blocks (names and corresponding instructions)": datapoint['assemgraph_com']['nodes'],
            "edges between two connected cfg_blocks": datapoint['assemgraph_com']['edges'], # (1,3)��ʾnodes��Ӧ���б��У���1���ڵ�͵�3���ڵ�֮����ڹ����ߣ��ڵ��������0��ʼ
            "cfg_block count": datapoint['assemgraph_com']['nodenum'] # �ú��������ܳ�ȡ������block������
            }
        cfg_prompt = json.dumps(cfg_prompt)
        prompt4input = Template("""You are a professional decompilation assistant. Please understand the following control flow graph and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code. Please ensure that the generated C source code is well-structured, syntactically correct, and reflects the accurate logic and data of the assembly code.

        - The control flow graph:
        ```Json
        $cfg_prompt
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, cfg_prompt=cfg_prompt, data_mapping=data_mapping)

    else:
        raise ValueError
    # print("there is OK after template")
    datapoint["data"] = json.dumps(datapoint["data"], ensure_ascii=False)
    datapoint["assemgraph_com"] = json.dumps(datapoint["assemgraph_com"], ensure_ascii=False)
    return rprompt4input




def build_instruct_v0(datapoint):
    # datapoint["data"] = json.loads(datapoint["data"])
    # datapoint["assemgraph_com"] = json.loads(datapoint["assemgraph_com"])

    add_info = {
        "instruction set architecture": datapoint['arch'],
        "bit width": datapoint['mode'],
        "compiler optimization level": datapoint['opts']}
    add_info = json.dumps(add_info)

    data_mapping = {
        "stack variables (size and relative offset)": datapoint['data']['param'],
        "read-only constants (size and value)": datapoint['data']['.rodata'],
        "initialized global/static data (size and initial value)": datapoint['data']['.data'],
        "uninitialized global/static data (size and count)": datapoint['data']['.bss'],
    }
    data_mapping = json.dumps(data_mapping)

    if datapoint['assemgraph_com']['nodenum'] == 1:
        # ����ڵ�Ϊ1��prompt,��ʾ��ida��������û��cfg graph
        assembly_code = datapoint['asm_ida_com'] # ida�����ķ������룬ֻ��һ��cfg block��û��ͼ

        prompt4input = Template("""You are a professional decompilation assistant. Within $add_info, please understand the following assembly code and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code. Please ensure that the generated C source code is well-structured, syntactically correct, and reflects the accurate logic and data of the assembly code.

        - The assembly code:
        ```Assembly
        $assembly_code
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, assembly_code=assembly_code, data_mapping=data_mapping, add_info=add_info)


    elif datapoint['assemgraph_com']['nodenum'] > 1:

        cfg_prompt = {
            "cfg_blocks (names and corresponding instructions)": datapoint['assemgraph_com']['nodes'],
            "edges between two connected cfg_blocks": datapoint['assemgraph_com']['edges'], # (1,3)��ʾnodes��Ӧ���б��У���1���ڵ�͵�3���ڵ�֮����ڹ����ߣ��ڵ��������0��ʼ
            "cfg_block count": datapoint['assemgraph_com']['nodenum'] # �ú��������ܳ�ȡ������block������
            }
        cfg_prompt = json.dumps(cfg_prompt)
        prompt4input = Template("""You are a professional decompilation assistant. Within $add_info, please understand the following control flow graph and data mapping table (defines the correspondence between data labels and their actual values), and perform decompilation into corresponding high-level C source code. Please ensure that the generated C source code is well-structured, syntactically correct, and reflects the accurate logic and data of the assembly code.

        - The control flow graph:
        ```Json
        $cfg_prompt
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitute(prompt4input, cfg_prompt=cfg_prompt, data_mapping=data_mapping, add_info=add_info)

    else:
        raise ValueError
    # print("there is OK after template")
    datapoint["data"] = json.dumps(datapoint["data"], ensure_ascii=False)
    datapoint["assemgraph_com"] = json.dumps(datapoint["assemgraph_com"], ensure_ascii=False)
    return rprompt4input


def old_build_instruct(datapoint):
    assembly_info = {
        "instruction set architecture": datapoint['arch'],
        "bit width": datapoint['mode'],
        "compiler optimization level": datapoint['opts']
    }
    assembly_prompt = datapoint['asm_ida_com'] if _ISIDA else datapoint['asm_obj']
    assembly_ida_con = datapoint['asm_ida_con']
    cfg_prompt = {
        "cfg_blocks": datapoint['assemgraph_com']['nodes'],
        "edges": datapoint['assemgraph_com']['edges'] if 'edges' in datapoint['assemgraph_com'] else [],
        "cfg_block count": datapoint['assemgraph_com']['nodenum']
    }
    cfg_con = {
        "cfg_blocks": datapoint['assemgraph_con']['nodes'],
        "edges": datapoint['assemgraph_con']['edges'] if 'edges' in datapoint['assemgraph_con'] else [],
        "cfg_block count": datapoint['assemgraph_con']['nodenum']
    }
    data_mapping = {
        "param": datapoint['data']['param'],
        ".rodata": datapoint['data']['.rodata'],
        ".data": datapoint['data']['.data'],
        ".bss": datapoint['data']['.bss'],
    }
    assembly_info = json.dumps(assembly_info)
    cfg_prompt = json.dumps(cfg_prompt)
    data_mapping = json.dumps(data_mapping)

    # asm_obj = datapoint['asm_obj']

    return Template.substitute(prompt4input, assembly_info=assembly_info, assembly_prompt=assembly_prompt, assembly_ida_con=assembly_ida_con, cfg_prompt=cfg_prompt, cfg_con=cfg_con, data_mapping=data_mapping)
    # return Template.substitute(prompt4input, cfg_prompt=cfg_prompt, data_mapping=data_mapping)
    # return Template.substitute(prompt4input, cfg_prompt=cfg_prompt, assembly_prompt=assembly_prompt)
    # return Template.substitute(prompt4input, asm_obj = asm_obj)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
llm = LLM(
    model=args.model_path,
    dtype="float16",
    tokenizer=args.model_path,
    trust_remote_code=True,
    tensor_parallel_size = 1, # ����Ϊ1��X�ſ�ΪX
    gpu_memory_utilization=0.5,
    disable_log_stats = True,
    disable_custom_all_reduce  = True,
    worker_use_ray = False,
)
# �������ɲ������ɸ���ԭ�в�����
sampling_params = SamplingParams(
    max_tokens=4096,  # ��Ӧԭ����max_new_tokens
    temperature=0.0,  # ����ȷ�������
    top_p=1.0,
    stop=[tokenizer.eos_token],  # ʹ��ԭtokenizer�Ľ�����
)
# # �ع�������ѭ��
# all_prompts = []
# task_mappings = []

print('Model Loaded!')

for ind in trange(start_index, end_index):
    file_path = os.path.join(input_path, f'func{ind}.jsonl')
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get("mode") == _BITWIDE:
                opts_value = json_obj.get("opts")
                input_asm_prompt_id = f"{ind}_{opts_value}"

                input_asm_prompt = build_instruct(datapoint=json_obj)
                outputs = llm.generate([input_asm_prompt], sampling_params)
                c_func_decompile = outputs[0].outputs[0].text.strip()
                json_obj['c_func_decompile'] = c_func_decompile

                jsonl_filename = os.path.join(result_path, f'func{ind}.jsonl')
                with open(jsonl_filename, 'a') as jsonl_file:
                    jsonl_file.write(json.dumps(json_obj) + '\n')
