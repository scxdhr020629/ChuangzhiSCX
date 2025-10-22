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

    # 系统提示词（增强版）
    system_prompt = """你是一个解决抽象推理语料库(ARC)谜题的专家。你擅长模式识别、逻辑推理和规则发现。

你的任务是：
1. 仔细分析训练样本中的所有变换模式
2. 识别在所有样本中一致的变换规则
3. 精确地应用这个规则来生成测试输出

请系统化、精确地进行分析，特别注意复杂的组合规则。"""

    # 用户提示词（增强版，包含更多复杂模式的引导）
    user_prompt = f"""请逐步解决这个ARC谜题。注意：这可能是一个复杂的多步骤变换。

## 训练样本：
{train_examples}

## 测试输入：
{test_input}

## 解题指南：

### 步骤1：深度观察分析
对每个训练样本进行全面分析：

**基础信息：**
- 输入和输出的精确维度（行数×列数）
- 维度变化规律（是否有扩展、缩减、或保持不变）
- 所有非零值的位置和数值
- 0值（背景）和非0值的分布模式

**结构分析：**
- 网格是否可以分割成多个子区域（如3×3块、行组、列组）
- 是否存在重复的模式或对称性
- 边界、角落、中心区域是否有特殊处理

### 步骤2：高级模式识别
检查以下复杂变换模式：

**颜色/数值变换：**
- 简单映射（如：1→2, 2→3）
- 条件映射（如：基于位置或邻居的颜色变换）
- 优先级规则（如：某些颜色覆盖其他颜色）

**空间变换：**
- 平移、旋转（90°、180°、270°）、镜像反射
- 缩放（放大或缩小）
- 区域选择（从多个块中选择特定块）
- 区域复制或重排

**组合变换：**
- 多步骤变换（先变换A，再变换B）
- 条件变换（如果条件X，则应用规则Y）
- 部分变换（只对特定区域或特定颜色应用规则）

**特殊模式：**
- 块选择规则（如：基于颜色优先级选择3×3块）
- 内容填充规则（如：复制部分内容到新区域）
- 模式延续（如：继续某个序列或图案）
- 计数和数学运算

### 步骤3：规则假设与验证
基于观察，形成规则假设：

1. **初步假设**：描述你观察到的主要模式
2. **详细规则**：精确定义变换步骤
3. **验证**：将规则应用到每个训练样本
   - 训练样本1：输入 → [应用规则] → 预期输出？ ✓/✗
   - 训练样本2：输入 → [应用规则] → 预期输出？ ✓/✗
   - 训练样本3：输入 → [应用规则] → 预期输出？ ✓/✗（如果有）

**重要**：如果规则在任何训练样本上失败，重新分析并调整规则。

### 步骤4：规则精炼
确认最终规则：
- 清晰地陈述完整的变换规则
- 如果是多步骤，列出每个步骤
- 确保规则能解释所有训练样本

### 步骤5：测试应用
将验证过的规则系统地应用到测试输入：
1. 显示每个变换步骤
2. 展示中间结果（如果有多步骤）
3. 生成最终输出

### 步骤6：最终输出
提供输出网格作为Python二维列表。

## 重要提示：
- 考虑复杂的多步骤变换
- 注意网格可能被分割成块进行处理
- 某些规则可能基于颜色优先级或条件判断
- 你的最终答案必须用"最终输出："标记
- 网格格式必须是有效的Python列表，如 [[1,2,3],[4,5,6]]
- 仔细验证输出的维度和数值

现在开始你的详细分析："""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages


def construct_prompt_with_hints(d, enable_hints=True):
    """
    构造带有额外提示的prompt（可选）

    这个版本会根据输入输出的特征给出更具体的提示
    """
    # 分析训练样本的特征
    input_shapes = [f"{len(ex['input'])}×{len(ex['input'][0])}" for ex in d['train']]
    output_shapes = [f"{len(ex['output'])}×{len(ex['output'][0])}" for ex in d['train']]

    # 检查是否有维度变化
    has_size_change = any(input_shapes[i] != output_shapes[i] for i in range(len(input_shapes)))

    # 检查是否可能有块结构
    possible_blocks = False
    for ex in d['train']:
        h, w = len(ex['input']), len(ex['input'][0])
        if (h % 3 == 0 or w % 3 == 0):
            possible_blocks = True
            break

    # 基础prompt
    messages = construct_prompt(d)

    if enable_hints:
        # 添加额外提示
        hints = "\n\n## 额外观察提示：\n"

        if has_size_change:
            hints += "- ⚠️ 注意：输入和输出的维度不同，可能涉及扩展、缩减或选择操作\n"

        if possible_blocks:
            hints += "- ⚠️ 网格维度可被3整除，考虑是否存在3×3块的处理模式\n"

        # 检查颜色数量
        all_values = set()
        for ex in d['train']:
            for row in ex['input'] + ex['output']:
                all_values.update(row)

        if len(all_values) <= 3:
            hints += f"- ⚠️ 只使用了{len(all_values)}种颜色值 {sorted(all_values)}，可能存在简单的颜色映射规则\n"

        # 将提示插入到用户消息中
        messages[1]['content'] = messages[1]['content'].replace(
            "现在开始你的详细分析：",
            hints + "\n现在开始你的详细分析："
        )

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


def parse_output(text, debug=False, sample_idx=None):
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