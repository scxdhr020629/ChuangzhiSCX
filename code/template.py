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
1. 仔细观察训练样本中的矩阵（全局、局部、分块）
2. 识别所有样本中一致的单步规则
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
仔细观察每个训练样本的矩阵，从以下角度入手：
- **全局视图**：整体网格维度（输入vs输出的大小）、非零值的分布模式、颜色/数值整体变换。
- **局部细节**：所有非零值的位置、相邻元素的交互、颜色/数值模式（0代表背景）。
- **分块观察**：将矩阵分成小块（如2x2或行/列分组），分析每个块内的形状、图案或变化；比较输入块与输出块的对应关系。
- 元素之间的空间关系、特殊的形状或图案。

### 步骤2：模式发现
基于观察，识别单一的变换规则。通过以下方面：
- 移动模式（平移、旋转、反射、缩放）
- 颜色/数值变换
- 元素的分组或分割
- 对称操作
- 数学关系
- 基于区域的操作（角落、边缘、中心）
- 复制或删除模式

**规则构建原则**：
- 假设一个简单的单步规则（如“所有输入元素顺时针旋转90度”），确保它在所有样本中一致。
- 优先最简单的解释（奥卡姆剃刀原则）。
- 清晰描述操作的细节。

### 步骤3：规则验证
清晰地陈述单步规则，并逐样本验证它适用于所有训练样本。
- 对于每个训练样本：描述输入 → 应用规则 → 预期输出 → 与实际输出比较。
- 该规则必须能够为每个训练输入生成精确的输出（完全匹配）。
- 如果规则不匹配，明确说明失败点，并调整为更合适的单步规则。

示例验证格式：
**样本1：**
- 输入：[...]
- 应用规则：旋转90度 → 输出：[...]
- 实际输出：[...]
- 匹配：是/否（如果否，说明差异）

### 步骤4：测试应用
系统地将验证过的单步规则应用到测试输入上。
- 描述测试输入。
- 展示推理过程和结果网格（用简要格式表示，如 [[1,0],[0,2]]）。

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


def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
    """

    # 策略1: 查找 "最终输出:" 或 "FINAL OUTPUT:" 标记
    patterns = [
        r"最终输出[：:]\s*\n?(.*?)(?:\n\n|\Z)",
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
    keywords = ['输出:', 'output:', 'result:', 'answer:', 'solution:',
                'prediction:', 'grid:', '答案:', '结果:', '预测:']
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