import json
from string import Template

def prompt1(datapoint):
    assembly_prompt = {
        "instruction set architecture": datapoint['arch'],
        "bit width": datapoint['mode'],
        "compiler optimization level": datapoint['opts'],
        "assembly code": datapoint['asm_ida_com']}

    data_mapping = {
        "stack variables (size and relative offset)": datapoint['data']['param'],
        "read-only constants (size and value)": datapoint['data']['.rodata'],
        "initialized global/static data (size and initial value)": datapoint['data']['.data'],
        "uninitialized global/static data (size and count)": datapoint['data']['.bss'],
    }
    assembly_prompt = json.dumps(assembly_prompt)
    data_mapping = json.dumps(data_mapping)

    if datapoint['assemgraph_com']['nodenum'] == 1:
        # 构造节点为1的prompt,表示由ida产生但是没有cfg graph
        assembly_prompt = datapoint['asm_ida_com'] # ida产生的反汇编代码，只有一个cfg block，没有图

        prompt4input = Template("""Please understand the function's logic and data based on the following assembly code and data mapping table (defines the correspondence between data labels and their actual values), and predict the corresponding high-level C source code.

        - The assembly code: 
        ```Json
        $assembly_prompt
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitue(prompt4input, assembly_prompt=assembly_prompt, data_mapping=data_mapping)


    elif datapoint['assemgraph_com']['nodenum'] > 1:
        cfg_prompt = {
            "cfg_blocks (names and corresponding instructions)": datapoint['assemgraph_com']['nodes'],
            "edges between two connected cfg_blocks": datapoint['assemgraph_com']['edges'], # (1,3)表示nodes对应的列表中，第1个节点和第3个节点之间存在关联边，节点的索引从0开始
            "cfg_block count": datapoint['assemgraph_com']['nodenum'] # 该函数体所能抽取出来的block的数量
            }
        cfg_prompt = json.dumps(cfg_prompt)
        prompt4input = Template("""Please understand the function's logic and data based on the following control flow graph and data mapping table (defines the correspondence between data labels and their actual values), and predict the corresponding high-level C source code.

        - The control flow graph:
        ```Json
        $cfg_prompt
        ```

        - The data mapping table:
        ```Json
        $data_mapping
        ```

        The high-level C source code is: """)
        rprompt4input = Template.substitue(prompt4input, cfg_prompt=cfg_prompt, data_mapping=data_mapping)

    else:
        raise ValueError

    return rprompt4input


