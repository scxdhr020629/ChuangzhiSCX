import json
import re
import ast


def construct_prompt(d):
    """
    构造用于大语言模型的提示词

    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。

    返回:
    list: OpenAI API的message格式列表
    """

    # 格式化训练样本
    train_examples = ""
    for i, example in enumerate(d['train'], 1):
        train_examples += f"\n### Training Example {i}:\n"
        train_examples += f"Input Grid:\n{format_grid(example['input'])}\n"
        train_examples += f"Output Grid:\n{format_grid(example['output'])}\n"

    # 格式化测试输入
    test_input = format_grid(d['test'][0]['input'])

    # 系统提示词
    system_prompt = """You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles. You excel at pattern recognition and logical reasoning.

Your task is to:
1. Analyze transformation patterns in training examples
2. Identify the consistent rule across ALL examples
3. Apply this rule precisely to generate the test output

Be systematic and precise in your analysis."""

    # 用户提示词 - 详细的推理指导
    user_prompt = f"""Solve this ARC puzzle step by step.

## Training Examples:
{train_examples}

## Test Input:
{test_input}

## Solution Process:

### Step 1: Detailed Observation
For each training example, note:
- Grid dimensions (input vs output)
- Position of all non-zero values
- Color/value patterns (0 is background)
- Spatial relationships between elements

### Step 2: Pattern Discovery
Identify the transformation by checking:
- Movement patterns (translation, rotation, reflection, scaling)
- Color/value transformations
- Grouping or splitting of elements
- Symmetry operations
- Mathematical relationships
- Region-based operations (corners, edges, center)

### Step 3: Rule Verification
State the rule clearly and verify it works for ALL training examples.
The rule MUST produce the exact output for each training input.

### Step 4: Test Application
Apply the verified rule to the test input systematically.

### Step 5: Final Output
Provide the output grid as a Python 2D list.

CRITICAL: 
- Your final answer must be marked with "FINAL OUTPUT:" followed by the grid
- The grid must be a valid Python list like [[1,2,3],[4,5,6]]
- Double-check dimensions and values

Begin your analysis:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages


def format_grid(grid):
    """
    将网格格式化为易读的字符串表示
    """
    lines = []
    for row in grid:
        row_str = ' '.join(str(cell).rjust(2) for cell in row)
        lines.append(row_str)
    return '\n'.join(lines)


def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    """

    # 策略1: 查找 "FINAL OUTPUT:" 标记
    patterns = [
        r"FINAL OUTPUT:\s*\n?(.*?)(?:\n\n|\Z)",
        r"Final Output:\s*\n?(.*?)(?:\n\n|\Z)",
        r"final output:\s*\n?(.*?)(?:\n\n|\Z)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            output_text = match.group(1).strip()
            grid = extract_grid_from_text(output_text)
            if grid and validate_grid(grid):
                return grid

    # 策略2: 查找所有Python列表格式
    list_pattern = r'\[\s*\[[\d,\s\[\]]+\]\s*\]'
    matches = re.findall(list_pattern, text)

    # 从后往前查找（最后一个通常是答案）
    for match in reversed(matches):
        try:
            # 清理并解析
            cleaned = re.sub(r'\s+', '', match)
            grid = ast.literal_eval(cleaned)

            if validate_grid(grid):
                # 转换为整数
                grid = [[int(cell) for cell in row] for row in grid]
                return grid
        except:
            continue

    # 策略3: 查找关键词后的网格
    keywords = ['output:', 'result:', 'answer:', 'solution:', 'prediction:', 'grid:']
    for keyword in keywords:
        if keyword in text.lower():
            idx = text.lower().rfind(keyword)  # 使用rfind找最后一次出现
            subset = text[idx:idx + 1000]
            grid = extract_grid_from_text(subset)
            if grid and validate_grid(grid):
                return grid

    # 策略4: 查找数字矩阵格式
    lines = text.split('\n')
    for i in range(len(lines) - 1, -1, -1):  # 从后往前搜索
        if re.match(r'^[\d\s,\[\]]+$', lines[i].strip()):
            # 收集连续的数字行
            grid_lines = []
            j = i
            while j >= 0 and re.match(r'^[\d\s,]+$', lines[j].strip()):
                if lines[j].strip():
                    grid_lines.insert(0, lines[j].strip())
                j -= 1

            if grid_lines:
                grid = parse_grid_lines(grid_lines)
                if grid and validate_grid(grid):
                    return grid

    # 如果都失败，返回默认值
    print(f"Warning: Could not parse valid grid from output text")
    return [[0]]


def extract_grid_from_text(text):
    """
    从文本中提取网格
    """
    # 尝试解析Python列表
    try:
        list_pattern = r'\[\s*\[[\d,\s\[\]]+\]\s*\]'
        match = re.search(list_pattern, text)
        if match:
            cleaned = re.sub(r'\s+', '', match.group(0))
            grid = ast.literal_eval(cleaned)
            if validate_grid(grid):
                return [[int(cell) for cell in row] for row in grid]
    except:
        pass

    # 尝试解析格式化的数字行
    lines = text.strip().split('\n')
    grid = parse_grid_lines(lines)
    if grid and validate_grid(grid):
        return grid

    return None


def parse_grid_lines(lines):
    """
    解析格式化的网格行
    """
    grid = []
    for line in lines:
        # 提取所有数字
        numbers = re.findall(r'\d+', line)
        if numbers:
            row = [int(n) for n in numbers]
            grid.append(row)

    if grid and validate_grid(grid):
        return grid

    return None


def validate_grid(grid):
    """
    验证网格是否有效
    """
    if not isinstance(grid, list) or len(grid) == 0:
        return False

    if not all(isinstance(row, list) for row in grid):
        return False

    # 检查所有行长度是否一致
    first_len = len(grid[0])
    if not all(len(row) == first_len for row in grid):
        return False

    # 检查所有元素是否可转换为整数
    try:
        for row in grid:
            for cell in row:
                int(cell)
        return True
    except:
        return False