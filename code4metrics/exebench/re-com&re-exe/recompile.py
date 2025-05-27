import ast, json, time, re, subprocess, os, argparse, shutil, math, traceback
from typing import Optional, Tuple
from LogRecorder import CLogRecoder
from string import Template
from tqdm import tqdm, trange
# import http.client
from openai import OpenAI, APIStatusError
import tree_sitter_cpp as tsc
from tree_sitter import Language, Parser

pro_path = '/exebench'
data_path = os.path.join(pro_path, 'origin')
input_path = ''
decom_flag = 'c_func_decompile'
_DEFAULT_CMD_TIMEOUT = 50
_INTERVAL = 100
_EXELEN = 5000
_KEY = '***'
_MODEL = 'gpt-4o'
_INDEX = '0'
_HANMODEL = 'deepseek-chat'
_GPTHD = False
_CFGTT = False
_DATATT = False
_IDATT = False

parser = argparse.ArgumentParser(description="Correctness of re-executablity")
parser.add_argument('--workdir', type=str, default=pro_path, required=False)
parser.add_argument('--data_path', type=str, default='origin', required=False)
parser.add_argument('--input', type=str, default='', required=False)
parser.add_argument('--model', type=str, default=_MODEL, required=False)
parser.add_argument('--key', type=str, default='', required=False)
parser.add_argument("-s", required=True, help="start")
parser.add_argument("-x", required=True, help="interval")
parser.add_argument("-b", required=True, help="bit-wide")
parser.add_argument("-i", required=True, help="index")
parser.add_argument("--gpt_handle", action="store_true", help="no post-handle")
parser.add_argument("--cfg_test", action="store_true", help="test block num > 1")
parser.add_argument("--data_test", action="store_true", help="test block num > 1")
parser.add_argument("--ida_test", action="store_true", help="test block num > 1")
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
if args.input:
    input_path = os.path.join(pro_path, args.input)
    if not os.path.exists(input_path):
        print(f'Error: input {input_path} not exists!')
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
if args.key:
    _KEY = args.key
if args.cfg_test:
    _CFGTT = True
if args.data_test:
    _DATATT = True
if args.ida_test:
    _IDATT = True
    decom_flag = 'pscode'
# is_first = True if args.f else False
if not args.s or not args.x:
    start_index = 0
    end_index = _EXELEN
    module = ''
else:
    start_index = int(args.s) * _INTERVAL
    end_index = start_index + int(args.x) * _INTERVAL
    module = '_{}-{}'.format(args.s, int(args.s) + int(args.x))
    if end_index > _EXELEN:
        end_index = _EXELEN
print("[+] {} to {} func".format(start_index, end_index))

source_path = os.path.join(data_path, 'source')
in_path = os.path.join(data_path, 'input')
out_path = os.path.join(data_path, 'output')
lib_path = os.path.join(pro_path, 'exebench')
res_path = os.path.join(pro_path, f'exebench_{_MODEL}_{_INDEX}')
analysis_path = os.path.join(pro_path, 'analysis')
if not input_path:
    input_path = os.path.join(res_path, f'decom_{_BITWIDE}')
tp_path = os.path.join(res_path, 'tmp')
tmp_path = os.path.join(tp_path, '{}{}{}'.format(_BITWIDE, module, ('_g' if _GPTHD else '')))
store_path = os.path.join(res_path, 'c_file_{}{}'.format(_BITWIDE, ('_gpthd' if _GPTHD else '')))
gpthd_path = os.path.join(res_path, 'gpthd_{}'.format(_BITWIDE))
if not os.path.exists(source_path):
    print(f'Error: source {source_path} not exists!')
    quit(-1)
if not os.path.exists(in_path):
    print(f'Error: standard input {in_path} not exists!')
    quit(-1)
if not os.path.exists(out_path):
    print(f'Error: standard output {out_path} not exists!')
    quit(-1)
if not os.path.exists(lib_path):
    print(f'Error: tmp dir {lib_path} not exists!')
    quit(-1)
if not os.path.exists(res_path):
    print(f'Error: result dir {res_path} not exists!')
    quit(-1)
if not os.path.exists(input_path):
    print(f'Error: decom dir {input_path} not exists!')
    quit(-1)
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)
if not os.path.exists(tp_path):
    os.makedirs(tp_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
# if os.path.exists(store_path):
#     shutil.rmtree(store_path)
# os.makedirs(store_path)
if not os.path.exists(store_path):
    os.makedirs(store_path)
if _GPTHD and not os.path.exists(gpthd_path):
    os.makedirs(gpthd_path)

changed_func = ['foo', 'getU1', 'ikcp_wndsize', 'main', 'push', 'reshape', 'Resize', 'split', 'strupper']
prefix = """
typedef unsigned long int uint64_t;
typedef long int int64_t;
typedef unsigned long int u64;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned int u32;
typedef short int int16_t;
typedef unsigned short int uint16_t;
typedef unsigned short int u16;
// typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned char u8;
typedef float float32;
"""

ymd = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logger = CLogRecoder(logfile=os.path.join(analysis_path, '{}_exe-metric_{}_{}{}_{}{}.log'.format(ymd, _MODEL, _BITWIDE, ('_gpthd' if _GPTHD else ''), _INDEX, module)))

C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)
os.environ['OPENAI_API_KEY'] = _KEY
os.environ['OPENAI_BASE_URL'] = '***'
os.environ["OPENAI_LOG"] = "false"
client = OpenAI()
prompt_file = os.path.join(pro_path, 'handle_prompt.txt')
if not os.path.exists(prompt_file):
    print(f'Error: no _prompt {prompt_file} not exists!')
    quit(-1)
with open(prompt_file, 'r') as f:
    prompt_template = Template(f.read())

def get_content(funcname, content):
    matches = re.findall(r'```.*?\n((.|\n)*)```', content, re.MULTILINE)
    if matches:
        content = matches[0][0].strip()
    else:
        content = content.strip()
    if _IDATT:
        content = content.replace(' __fastcall ', ' ')
        content = content.replace(' __cdecl ', ' ')
    return content

# def handle_decom(funcname, content):
#     lines = content.split('\n')
#     start_index = 0
#     for index in range(len(lines)):
#         if lines[index]:
#             mat = re.findall(r'([a-zA-Z0-9_]{2,})[ ]*\(', lines[index], re.MULTILINE)
#             if mat:
#                 changed = ' ' + funcname + '('
#                 line = re.sub(r'([a-zA-Z0-9_]{2,})[ ]*\(', changed, lines[index], re.MULTILINE)
#                 if line != lines[index]:
#                     lines[index] = line
#                     logger.INFO("changed decom {}".format(funcname))
#                 start_index = index
#                 break
#     if index == len(lines) - 1:
#         logger.INFO("No change decom {}".format(funcname))
#     return '\n'.join(lines[index:])

def handle_decom(decom: str, funcname: str) -> str:
    tree = parser.parse(bytes(decom, "utf8"))
    root_node = tree.root_node
    code_bytes = bytearray(decom, "utf8")

    output = []
    query = C_LANGUAGE.query(" (declaration (init_declarator((number_literal)))) @true_value")
    captures = query.captures(root_node)
    if captures:
        for capture in captures.values():
            for value in capture:
                output.append(value.text.decode("utf8"))

    call_function = []
    query = C_LANGUAGE.query('(call_expression function: (identifier) @func_name)')
    captures = query.captures(root_node)
    if captures:
        for key, capture in captures.items():
            for value in capture:
                call_function.append(value.text.decode("utf8"))
    query = C_LANGUAGE.query(" (declaration ( pointer_declarator (function_declarator(_)  @declar_value1 ))) ")
    captures = query.captures(root_node)
    if captures:
        for key, capture in captures.items():
            for value in capture:
                call_function.append(value.text.decode("utf8"))
    query = C_LANGUAGE.query(" (declaration ( function_declarator(_)  @declar_value2 ))")
    captures = query.captures(root_node)
    if captures:
        for key, capture in captures.items():
            for value in capture:
                call_function.append(value.text.decode("utf8"))
    
    last_functions = []
    changes = []
    changes_filter = []
    function_filter = []
    def walk(node):
        # nonlocal last_function
        # nonlocal changes
        # if node.type in ['function_definition','function_declaration']:
        if node.type in ['function_definition']:
            for child in node.children:
                if child.type == 'function_declarator':
                    for c_child in child.children:
                        if c_child.type == 'identifier':
                            if c_child.text.decode("utf8") not in call_function:
                                changes.append((c_child.start_byte, c_child.end_byte, c_child.text.decode("utf8")))
                                last_functions.append(node)
                            else:
                                changes_filter.append((c_child.start_byte, c_child.end_byte, c_child.text.decode("utf8")))
                                function_filter.append(node)
                elif child.type == 'pointer_declarator':
                    for c_child in child.children:
                        if c_child.type == 'function_declarator':
                            for cc_child in c_child.children:
                                if cc_child.type == 'identifier':
                                    if cc_child.text.decode("utf8") not in call_function:
                                        changes.append((cc_child.start_byte, cc_child.end_byte, cc_child.text.decode("utf8")))
                                        last_functions.append(node)
                                    else:
                                        changes_filter.append((cc_child.start_byte, cc_child.end_byte, cc_child.text.decode("utf8")))
                                        function_filter.append(node)
        for child in node.children:
            walk(child)

    walk(root_node)
    # changes.sort(reverse=True)
    if len(changes) == 1:
        change = changes[-1]
        last_function = last_functions[-1]
        function_text = code_bytes[last_function.start_byte:last_function.end_byte]
        start_byte, end_byte = (change[0] - last_function.start_byte), (change[1] - last_function.start_byte)
        if start_byte <= 0 or start_byte >= len(function_text) or end_byte <= 0 or end_byte >= len(function_text):
            logger.INFO("start_byte error ({}, {}) ({}, {}) <= 0 {}".format(change[0], change[1], last_function.start_byte, last_function.end_byte, funcname))
        else:
            function_text = function_text[:start_byte] + bytearray(funcname, "utf8") + function_text[end_byte:]
        output.append(function_text.decode("utf8"))
    else:
        if not changes and not changes_filter:
            logger.INFO("Error changes and filter {}\ncall_function {}: {}".format(funcname, len(call_function), call_function))
        else:
            if not changes:
                changes = changes_filter
                last_functions = function_filter
                logger.INFO("Choose filter {}\nchanges_filter: {}".format(funcname, changes_filter))
            else:
                logger.INFO("Choose changes {}\nchanges: {}".format(funcname, changes))
            max_len = 0
            chosen_ind = -1
            for ind in range(len(last_functions)):
                len_t = last_functions[ind].start_byte - last_functions[ind].end_byte
                if len_t > max_len:
                    max_len = len_t
                    chosen_ind = ind
            change = changes[chosen_ind]
            last_function = last_functions[chosen_ind]
            function_text = code_bytes[last_function.start_byte:last_function.end_byte]
            start_byte, end_byte = (change[0] - last_function.start_byte), (change[1] - last_function.start_byte)
            if start_byte <= 0 or start_byte >= len(function_text) or end_byte <= 0 or end_byte >= len(function_text):
                logger.INFO("start_byte error ({}, {}) ({}, {}) <= 0 {}".format(change[0], change[1], last_function.start_byte, last_function.end_byte, funcname))
            else:
                function_text = function_text[:start_byte] + bytearray(funcname, "utf8") + function_text[end_byte:]
            output.append(function_text.decode("utf8"))
    return '\n'.join(output)
    
def delete_conflict(funcname, c_deps, func_def):
    matches = re.findall(r'(\w+)__bench[ ]*\(.*?\)[ ]*{', c_deps, re.MULTILINE)
    # print(matches)
    if len(matches) > 0:
        # c_deps_0 = c_deps
        func_def_0 = func_def
        for item in matches:
            # pattern = r'^.*?(?<![a-zA-Z0-9_])' + item + r'(?![a-zA-Z0-9_]).*?;.*?$'
            # c_deps_1 = re.sub(pattern, '', c_deps_0, flags=re.M)
            # if len(c_deps_1) == len(c_deps_0):
            #     logger.INFO("Delete c_deps failed {} {}".format(item, funcname))
            # else:
            #     logger.INFO("Delete c_deps success {} {}".format(item, funcname))
            pattern = r'(?<![a-zA-Z0-9_])' + item + r'(?![a-zA-Z0-9_])[ ]*\('
            changed = item + '__bench('
            func_def_1 = re.sub(pattern, changed, func_def_0, flags=re.M)
            if len(func_def_1) == len(func_def_0):
                logger.INFO("Delete func_def failed {} {}".format(item, funcname))
            # else:
            #     logger.INFO("Delete func_def success {} {}".format(item, funcname))
            # c_deps_0 = c_deps_1
            func_def_0 = func_def_1
        # c_deps = c_deps_0
        func_def = func_def_0
    # return c_deps, func_def
    return func_def

def call_with_messages(funcname, prompt, temperature, top_p):
    func_modified_file = os.path.join(gpthd_path, f'{funcname}.c')
    if os.path.exists(func_modified_file):
        with open(func_modified_file, 'r') as f:
            content = f.read()
        data = content
    else:
        for i in range(5):
            try:
                chat_completion = client.chat.completions.create(
                    model= _HANMODEL,
                    messages= [
                        {
                            "role": "user",
                            "temperature": temperature,
                            "top_p": top_p,
                            "content": prompt
                        }
                    ]
                )
                break
            except APIStatusError as e:
                if e.status_code in [408, 500, 400]:
                    print("ERROR: {}\n {}".format(funcname, traceback.format_exc()))
                    logger.INFO("Openai error {}\n {}".format(funcname, traceback.format_exc()))
                    return ""
                elif e.status_code == 429:
                    time.sleep(10)
                    if i == 4:
                        print("ERROR: {}\n {}".format(funcname, traceback.format_exc()))
                        logger.INFO("Openai error {}\n {}".format(funcname, traceback.format_exc()))
                        return ""
                    continue
                else:
                    print("ERROR: {}\n {}".format(funcname, traceback.format_exc()))
                    raise e

        if not hasattr(chat_completion, "choices") or not chat_completion.choices:
            print("Request failed!")
            print(chat_completion)
            logger.INFO("Request failed {}".format(funcname))
            return ""
        data = chat_completion.choices[0].message.content
        with open(func_modified_file, 'w') as f:
            f.write(data)
    
    # print(data)
    # logger.INFO("Gpt answer {}:\n{}".format(funcname, data))
    optimized = re.search(r"# optimized code[ ]*\n((.|\n)*?)#", data, re.I)
    if optimized:
        optimized_c = re.search(r"```C\n((.|\n)*)```", optimized[1], re.I)
        if optimized_c:
            optimized_code = optimized_c[1]
        else:
            optimized_code = ""
            logger.INFO("No optimized_code {}".format(funcname))
    else:
        optimized_code = ""
        logger.INFO("No match optimized {}".format(funcname))
    return optimized_code

def diff_io(observed_output, expected_output) -> bool:
    if type(observed_output) is not type(expected_output):
        return False
    if isinstance(observed_output, list):
        if len(observed_output) != len(expected_output):
            return False
        for e1, e2 in zip(observed_output, expected_output):
            ok = diff_io(e1, e2)
            if not ok:
                return False
    elif isinstance(observed_output, dict):
        for key in observed_output:
            if key not in expected_output:
                return False
            ok = diff_io(observed_output[key], expected_output[key])
            if not ok:
                return False
    elif isinstance(observed_output, float):
        ok = math.isclose(observed_output, expected_output)
        if not ok:
            return False
    else:
        ok = observed_output == expected_output
        if not ok:
            return False
    return True

def _run_command(command: str, timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[str, str]:
    # output = subprocess.run(command.split(), shell=True, capture_output=True, text=True, input=stdin, timeout=timeout)
    output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    stdout = output.stdout.decode('utf-8') if isinstance(output.stdout, bytes) else output.stdout
    stderr = output.stderr.decode('utf-8') if isinstance(output.stderr, bytes) else output.stderr
    return stdout, stderr

funcname_file = os.path.join(data_path, 'funcnames.txt')
if not os.path.exists(funcname_file):
    print(f'Error: no funcname file {funcname_file} not exists!')
    quit(-1)
with open(funcname_file, 'r') as f:
    func_names = f.read()
    func_names = ast.literal_eval(func_names)
func_names = func_names[start_index:end_index]
NUM = {"O0":0, "O1":0, "O2":0, "O3":0}
num_compile = {"O0":0, "O1":0, "O2":0, "O3":0}
num_run = {"O0":0, "O1":0, "O2":0, "O3":0}
num_idano = {"O0":0, "O1":0, "O2":0, "O3":0}
success_compile = {"O0": [], "O1": [], "O2": [], "O3": []}
success_run = {"O0": [], "O1": [], "O2": [], "O3": []}
c_file_tmp = os.path.join(tmp_path, 'combile.c')
cpp_wrapper_tmp = os.path.join(tmp_path, 'combile.cpp')
executable_onlyfunc = os.path.join(tmp_path, 'combile.x')
executable = os.path.join(tmp_path, 'combile')
output_tmp = os.path.join(tmp_path, 'output.json')

for ind in trange(len(func_names)):
    func = func_names[ind]
    # print('[+] {} {}'.format(ind, func))

    c_deps_file = os.path.join(source_path, f'{func}_{_BITWIDE}_deps.c')
    with open(c_deps_file, 'r') as f:
        c_deps = f.read()
    # if _BITWIDE == '32':
    #     c_deps = prefix + c_deps
    cpp_wrapper_file = os.path.join(source_path, f'{func}_{_BITWIDE}.cpp')
    with open(cpp_wrapper_file, 'r') as f:
        cpp_wrapper = f.read()
    cpp_wrapper = re.sub(r'(extern "C" \n\{\n#include ").*?("\n\})', r'\1' + c_file_tmp + r'\2', cpp_wrapper, re.M)
    with open(cpp_wrapper_tmp, 'w') as f:
        f.write(cpp_wrapper)
    
    file_input = os.path.join(in_path, f'{func}.json')
    file_output = os.path.join(out_path, f'{func}.json')
    with open(file_output, 'r') as f:
        output_stand = json.load(f)

    file_de = os.path.join(input_path, f'{func}.jsonl')
    if not os.path.exists(file_de):
        continue
    with open(file_de, 'r') as f:
        content = f.readlines()
    for ite in content:
        if not ite:
            continue
        item = json.loads(ite)
        if item["mode"] != _BITWIDE:
            continue
        if _CFGTT and item['assemgraph_con']['nodenum'] == 1:
            continue
        item['data'] = json.loads(item['data'])
        if _DATATT and not item['data']['.rodata'] and not item['data']['.data']:
            continue
        target = '{}_{}'.format(func, item["opts"])
        NUM[item["opts"]] += 1
        logger.INFO('{} {}'.format(ind, target))
        decom = get_content(target, item[decom_flag])
        is_compile = False

        cpp_store = os.path.join(store_path, f'{func}.cpp')
        if not os.path.exists(cpp_store):
            with open(cpp_store, 'w') as f:
                f.write(cpp_wrapper)
        c_func = decom
        c_origin_store = os.path.join(store_path, f'{target}_orig.c')
        with open(c_origin_store, 'w') as f:
            f.write(c_func)
        if not decom:
            logger.INFO('No origin decom {}'.format(target))
            if _IDATT:
                num_idano[item["opts"]] += 1
            continue
        
        with open(c_file_tmp, 'w') as f:
            f.write(c_func)
        if os.path.exists(executable_onlyfunc):
            os.remove(executable_onlyfunc)
        if os.path.exists(executable):
            os.remove(executable)
        if os.path.exists(output_tmp):
            os.remove(output_tmp)

        compile_command = 'gcc {} -fpermissive -o {} -c {}'.format(('-m32' if _BITWIDE == '32' else ''), executable_onlyfunc, c_file_tmp)
        try:
            stdout, stderr = _run_command(compile_command)
        except Exception as e:
            logger.INFO('Cmd error {}:\n{}'.format(target, e))
        if not os.path.exists(executable_onlyfunc):
            logger.INFO('No binary {}'.format(target))
            if stderr:
                logger.INFO('Compile error {}:\n{}'.format(target, stderr))
            # continue
        else:
            is_compile = True
            num_compile[item["opts"]] += 1
            success_compile[item["opts"]].append(func)

        try:
            decom = handle_decom(decom, func)
        except Exception:
            print("ERROR: {}\n {}".format(target, traceback.format_exc()))
            logger.INFO("Openai error {}\n {}".format(target, traceback.format_exc()))
        if not decom:
            logger.INFO('No decom {}'.format(target))
            continue
        decom = delete_conflict(target, c_deps, decom)
        if _GPTHD:
            func_orig_file = os.path.join(gpthd_path, f'{target}_orig.c')
            if not os.path.exists(func_orig_file):
                with open(func_orig_file, 'w') as f:
                    f.write(item['sourcecode'])
                    f.write('\n\n\n -----------------------------------------------------------\n')
                    f.write(decom)
            prompt = Template.substitute(prompt_template, dependency=c_deps, decompiled=decom)
            optimized_code = call_with_messages(target, prompt, 0, 1.0)
            if optimized_code:
                if 'none' not in optimized_code:
                    decom = optimized_code
        
        c_func = c_deps + '\n' + decom
        with open(c_file_tmp, 'w') as f:
            f.write(c_func)
        c_file_store = os.path.join(store_path, f'{target}.c')
        with open(c_file_store, 'w') as f:
            f.write(c_func)
        if os.path.exists(executable_onlyfunc):
            os.remove(executable_onlyfunc)

        compile_command = 'gcc {} -fpermissive -o {} -c {}'.format(('-m32' if _BITWIDE == '32' else ''), executable_onlyfunc, c_file_tmp)
        try:
            stdout, stderr = _run_command(compile_command)
        except Exception as e:
            logger.INFO('Cmd error {}:\n{}'.format(target, e))
        if not os.path.exists(executable_onlyfunc):
            logger.INFO('No binary {}'.format(target))
            if stderr:
                logger.INFO('Compile error {}:\n{}'.format(target, stderr))
            continue
        elif not is_compile:
            num_compile[item["opts"]] += 1
            success_compile[item["opts"]].append(func)

        compile_command = 'g++ {} -fpermissive -o {} {} -I {} -I{}'.format(('-m32' if _BITWIDE == '32' else ''), executable, cpp_wrapper_tmp, lib_path, lib_path)
        try:
            stdout, stderr = _run_command(compile_command)
        except Exception as e:
            logger.INFO('Cmd error {}:\n{}'.format(target, e))
        if not os.path.exists(executable):
            logger.INFO('No binary {}'.format(target))
            if stderr:
                logger.INFO('Compile cpp error {}:\n{}'.format(target, stderr))
            continue

        run_command = f'{executable} {file_input} {output_tmp}'
        try:
            process = subprocess.run(run_command, shell=True, check=True,capture_output=True, timeout=_DEFAULT_CMD_TIMEOUT)
        except subprocess.CalledProcessError as e:
            pass
        except Exception as e:
            pass
        if os.path.exists(output_tmp):
            with open(output_tmp, 'r') as f:
                content = f.read()
            output_store = os.path.join(store_path, f'{target}.json')
            with open(output_store, 'w') as f:
                f.write(content)
            if content:
                output = json.loads(content)
            else:
                output = content
            if diff_io(output, output_stand):
                num_run[item["opts"]] += 1
                success_run[item["opts"]].append(func)
            else:
                logger.INFO('Incorrect result {}:\n{}'.format(target, output))
                logger.INFO('Expected result {}:\n{}'.format(target, output_stand))

success_compile_file = os.path.join(store_path, f'compile_success{module}.json')
with open(success_compile_file, 'w') as f:
    json.dump(success_compile, f, ensure_ascii=True)
success_run_file = os.path.join(store_path, f'reexecu_success{module}.json')
with open(success_run_file, 'w') as f:
    json.dump(success_run, f, ensure_ascii=True)
metric_file = os.path.join(res_path, 'metric.txt')
with open(metric_file, 'a') as f:
    new_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    f.write('{} {} {} {}-{} {}\n'.format(new_time, _MODEL, _BITWIDE, start_index, end_index, ('GPTHD' if _GPTHD else '')))
    for opt_state in num_compile.keys():
        f.write('opt:{} count:{} compile_success:{} compile_rate:{:.4f} run_success:{} run_rate:{:.4f}\n'.format(opt_state, NUM[opt_state], num_compile[opt_state], num_compile[opt_state]/NUM[opt_state], num_run[opt_state], num_run[opt_state]/NUM[opt_state]))
    if _IDATT:
        f.write(json.dumps(num_idano))
        f.write('\n')
    f.write('\n')

