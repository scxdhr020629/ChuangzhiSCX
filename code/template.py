import json
import re
import ast


def construct_prompt(d):
    """
    构造用于大语言模型的提示词（增强版 - 包含完整的ARC操作类型）

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

    # 用户提示词（中文版 - 增强版）
    user_prompt = f"""请逐步解决这个ARC谜题。

## 训练样本：
{train_examples}

## 测试输入：
{test_input}

## 解题步骤：

### 步骤1：详细观察和操作类型初步识别

对每个训练样本，注意以下要点：
- 网格维度（输入vs输出的大小）
- 所有非零值的位置
- 颜色/数值模式（0代表背景）
- 元素之间的空间关系
- 特殊的形状或图案

**同时，初步识别这个任务可能涉及的操作类型（可多选）：**

**基础操作：**
- [ ] 简单颜色/数值映射 - 将一种颜色变成另一种颜色
- [ ] 平移(Translation) - 元素在网格中移动位置
- [ ] 旋转(Rotation) - 元素或整个网格旋转(90°/180°/270°)
- [ ] 翻转/反射(Reflection) - 沿水平/竖直/对角线翻转
- [ ] 缩放(Scaling) - 放大或缩小元素或整个网格

**对象和结构操作：**
- [ ] 对象提取/分离 - 提取特定颜色或形状的对象
- [ ] 连通分量识别 - 找出相邻的同色元素组作为一个整体
- [ ] 边界/框架操作 - 添加、删除或检测边框/轮廓
- [ ] 元素的分组或分割 - 根据位置或属性分组

**模式和重复操作：**
- [ ] 重复/平铺(Repetition) - 小图案重复排列
- [ ] 模式检测 - 识别重复出现的图案或周期性结构
- [ ] 复制模式 - 将模式从一个位置复制到另一个位置

**填充和区域操作：**
- [ ] 填充/清空(Fill/Erase) - 用特定颜色填充或清空区域
- [ ] 洪泛填充(Flood Fill) - 填充连续的同色区域
- [ ] 基于区域的操作 - 针对角落、边缘、中心等特定区域的操作
- [ ] 网格划分 - 将网格分成块或象限分别处理

**高阶逻辑操作：**
- [ ] 对称操作 - 基于对称性的变换
- [ ] 条件规则 - 根据条件应用不同的变换(如：如果在边界→变红)
- [ ] 逻辑AND/OR/NOT - 基于多个条件的布尔操作
- [ ] 数学关系 - 基于计数、距离、大小等数值关系

**尺寸和维度操作：**
- [ ] 网格扩展/扩大 - 增大网格尺寸
- [ ] 网格裁剪/缩小 - 减小网格尺寸
- [ ] 行/列重复 - 重复特定的行或列
- [ ] 行/列删除 - 删除特定的行或列
- [ ] 投影操作 - 将二维信息压缩到一维(横向/纵向投影)

**比较和关系操作：**
- [ ] 计数操作 - 基于元素数量的变换
- [ ] 相对位置操作 - 基于相对位置(左边/右边/上方/下方)
- [ ] 相似性检测 - 找相似或相同的元素
- [ ] 最大值/最小值提取 - 提取最大或最小的对象

**多步骤操作：**
- [ ] 多步序列 - 多个操作按顺序应用
- [ ] 变换链 - 一个操作的输出是下一个的输入
- [ ] 递归操作 - 对每个找到的对象重复同一操作

---

### 步骤2：模式发现（深度分析）

基于步骤1的初步识别，现在进行深度分析来识别变换规则。

**对每个训练样本进行详细的模式分析：**

**样本{1}分析：**

1. **基础信息**
   - 输入维度：___ × ___
   - 输出维度：___ × ___
   - 尺寸是否改变？(是/否)

2. **颜色/数值分析**
   - 输入中出现的所有非零颜色：___
   - 输出中出现的所有颜色：___
   - 颜色映射关系（如 1→2, 3→5）：___

3. **位置和空间分析**
   - 输入中非零元素的具体位置：___
   - 输出中非零元素的具体位置：___
   - 位置是否改变？如果改变，是什么类型的改变？(平移/旋转/翻转/其他)

4. **对象和结构分析**
   - 输入中是否有明确的"对象"(连续的非零元素)？
   - 如果有，有多少个对象？每个对象的形状是什么？
   - 输出中对象的数量和形状是否改变？

5. **模式和重复分析**
   - 输入中是否有重复的模式或周期性结构？
   - 输出中是否有重复的模式？
   - 重复的单元是什么？重复了多少次？

6. **特殊操作检查**
   - 是否涉及边框或轮廓？(添加/删除/改变)
   - 是否涉及填充操作？
   - 是否涉及对称性？
   - 是否有明确的条件规则(如：背景变为X，对象变为Y)？

7. **变换总结**
   用一句话总结该样本的主要变换：___

**样本2分析：** (如果有)
[重复上述分析流程]

**样本3+分析：** (如果有)
[重复上述分析流程]

**跨样本规律对比**
- 所有样本中的变换是否一致？
- 如果有差异，这些差异说明了什么？(是否有条件规则？)
- 最小公共规则是什么？

---

### 步骤3：规则验证
清晰地陈述规则，并验证它适用于所有训练样本。
该规则必须能够为每个训练输入生成精确的输出。

**最终规则陈述：**
"___"

**验证过程：**
- 规则是否在样本1上成立？(是/否) 如果否，差异是什么？
- 规则是否在样本2上成立？(是/否) 如果否，差异是什么？
- 规则是否在样本3+上成立？(是/否) 如果否，差异是什么？

如果有不一致，请调整规则并重新验证。

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