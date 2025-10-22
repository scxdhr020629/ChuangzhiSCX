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
        train_examples += f"\n### 训练样本 {i}:\n"
        train_examples += f"输入网格:\n{format_grid(example['input'])}\n"
        train_examples += f"输出网格:\n{format_grid(example['output'])}\n"

    # 格式化测试输入
    test_input = format_grid(d['test'][0]['input'])

    # 系统提示词（中文版）
    system_prompt = """你是一个解决抽象推理语料库(ARC)谜题的专家。你擅长模式识别和逻辑推理。

你的任务是：
1. 分析训练样本中的变换模式
2. 识别所有样本中一致的规则
3. 精确地应用这个规则来生成测试输出

请系统化和精确地进行分析。"""

    # 用户提示词（中文版）
    user_prompt = f"""请逐步解决这个ARC谜题。

## 训练样本：
{train_examples}

## 测试输入：
{test_input}

## 解题步骤：

### 步骤1：详细观察
对每个训练样本，注意以下要点：
- 网格维度（输入vs输出的大小）
- 所有非零值的位置
- 颜色/数值模式（0代表背景）
- 元素之间的空间关系
- 特殊的形状或图案

### 步骤2：模式发现
通过以下方面识别变换规则：
- 移动模式（平移、旋转、反射、缩放）
- 颜色/数值变换
- 元素的分组或分割
- 对称操作
- 数学关系
- 基于区域的操作（角落、边缘、中心）
- 复制或删除模式

### 步骤3：规则验证
清晰地陈述规则，并验证它适用于所有训练样本。
该规则必须能够为每个训练输入生成精确的输出。

### 步骤4：测试应用
系统地将验证过的规则应用到测试输入上。
展示每一步的推理过程。

### 步骤5：最终输出
将输出网格作为Python二维列表提供。

重要提示：
- 你的最终答案必须用"最终输出："标记，后面跟着网格
- 网格必须是有效的Python列表，如 [[1,2,3],[4,5,6]]
- 仔细检查维度和数值
- 确保输出格式正确

现在开始你的分析："""

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


def parse_output(text, debug=True, sample_idx=None):
    """
    解析大语言模型的输出文本，提取预测的网格

    参数:
    text (str): 大语言模型在设计prompt下的输出文本
    debug (bool): 是否输出调试信息
    sample_idx (int): 当前样本索引（用于调试输出）

    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    """

    sample_info = f"[Sample {sample_idx}]" if sample_idx else ""

    if debug:
        print(f"  {sample_info} Parsing output using multiple strategies...")

    # 策略1: 查找 "最终输出:" 或 "FINAL OUTPUT:" 标记
    if debug:
        print(f"  {sample_info} Strategy 1: Looking for '最终输出:' or 'FINAL OUTPUT:' markers...")

    patterns = [
        (r"最终输出[：:]\s*\n?(.*?)(?:\n\n|\Z)", "最终输出"),
        (r"FINAL OUTPUT:\s*\n?(.*?)(?:\n\n|\Z)", "FINAL OUTPUT"),
        (r"Final Output:\s*\n?(.*?)(?:\n\n|\Z)", "Final Output"),
        (r"final output:\s*\n?(.*?)(?:\n\n|\Z)", "final output")
    ]

    for pattern, pattern_name in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            if debug:
                print(f"    {sample_info} Found '{pattern_name}' marker!")
            output_text = match.group(1).strip()
            grid = extract_grid_from_text(output_text)
            if grid and validate_grid(grid):
                if debug:
                    print(f"    {sample_info} ✓ Successfully parsed grid using Strategy 1 (marker: {pattern_name})")
                return grid
            elif debug:
                print(f"    {sample_info} Found marker but failed to extract valid grid")

    # 策略2: 查找所有Python列表格式
    if debug:
        print(f"  {sample_info} Strategy 2: Looking for Python list format [[...]]...")

    list_pattern = r'\[\s*\[[\d,\s\[\]]+\]\s*\]'
    matches = re.findall(list_pattern, text)

    if matches:
        if debug:
            print(f"    {sample_info} Found {len(matches)} potential Python lists")

        # 从后往前查找（最后一个通常是答案）
        for i, match in enumerate(reversed(matches)):
            try:
                # 清理并解析
                cleaned = re.sub(r'\s+', '', match)
                grid = ast.literal_eval(cleaned)

                if validate_grid(grid):
                    # 转换为整数
                    grid = [[int(cell) for cell in row] for row in grid]
                    if debug:
                        print(
                            f"    {sample_info} ✓ Successfully parsed grid using Strategy 2 (list {len(matches) - i}/{len(matches)} from end)")
                    return grid
            except Exception as e:
                if debug and i == 0:  # 只对最后一个列表显示错误
                    print(f"    {sample_info} Failed to parse last list: {str(e)[:50]}")
                continue
    elif debug:
        print(f"    {sample_info} No Python list format found")

    # 策略3: 查找关键词后的网格
    if debug:
        print(f"  {sample_info} Strategy 3: Looking for grids after keywords...")

    keywords = ['输出:', 'output:', 'result:', 'answer:', 'solution:',
                'prediction:', 'grid:', '答案:', '结果:', '预测:']

    found_keywords = [kw for kw in keywords if kw in text.lower()]
    if found_keywords and debug:
        print(f"    {sample_info} Found keywords: {found_keywords}")

    for keyword in keywords:
        if keyword in text.lower():
            idx = text.lower().rfind(keyword)  # 使用rfind找最后一次出现
            subset = text[idx:idx + 1000]
            grid = extract_grid_from_text(subset)
            if grid and validate_grid(grid):
                if debug:
                    print(f"    {sample_info} ✓ Successfully parsed grid using Strategy 3 (keyword: '{keyword}')")
                return grid

    if debug and found_keywords:
        print(f"    {sample_info} Found keywords but failed to extract valid grid")

    # 策略4: 查找数字矩阵格式
    if debug:
        print(f"  {sample_info} Strategy 4: Looking for number matrix format...")

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
                if debug:
                    print(f"    {sample_info} Found {len(grid_lines)} lines of numbers")
                grid = parse_grid_lines(grid_lines)
                if grid and validate_grid(grid):
                    if debug:
                        print(f"    {sample_info} ✓ Successfully parsed grid using Strategy 4 (matrix format)")
                    return grid
                elif debug:
                    print(f"    {sample_info} Found number lines but failed to form valid grid")

    # 如果都失败，返回默认值
    if debug:
        print(f"  {sample_info} ✗ All strategies failed! Returning default [[0]]")
    else:
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