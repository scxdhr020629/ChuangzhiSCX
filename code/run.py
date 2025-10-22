import json
import argparse
from openai import OpenAI
from template import construct_prompt, parse_output


def calculate_accuracy(predicted, ground_truth):
    """
    计算完全匹配的准确率
    """
    if len(predicted) != len(ground_truth):
        return 0

    if any(len(pred_row) != len(gt_row) for pred_row, gt_row in zip(predicted, ground_truth)):
        return 0

    for i in range(len(predicted)):
        for j in range(len(predicted[i])):
            if predicted[i][j] != ground_truth[i][j]:
                return 0

    return 1


def main():
    # 直接在代码中设置参数
    API_KEY = "sk-32537d6922344b1faa9f853e971b23d4"  # 替换为你的API密钥
    BASE_URL = "https://api.deepseek.com"  # OpenAI API endpoint
    MODEL_NAME = "deepseek-chat"  # 或 "gpt-3.5-turbo"
    TEMPERATURE = 1.0  # 低温度以获得确定性输出
    NUM_EPOCHS = 1  # 评估轮数

    # 初始化OpenAI客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 读取验证集
    print("Loading validation dataset...")
    with open("val.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    samples = [json.loads(line) for line in lines]

    print(f"Loaded {len(samples)} samples")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 50)

    epoch_accuracies = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        correct = 0
        total = 0

        for idx, sample in enumerate(samples, start=1):
            # 保存ground truth
            ground_truth = sample['test'][0]['output']

            # 移除test中的output用于测试
            sample_input = sample.copy()
            sample_input['test'] = [{'input': sample['test'][0]['input']}]

            # 构造prompt
            messages = construct_prompt(sample_input)

            try:
                # 调用API
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=8000
                )

                # 获取输出文本
                output_text = response.choices[0].message.content.strip()

                # 解析输出
                predicted = parse_output(output_text)

                # 计算准确率
                is_correct = calculate_accuracy(predicted, ground_truth)
                correct += is_correct
                total += 1

                # 显示进度
                accuracy_so_far = correct / total
                status = "✓" if is_correct else "✗"
                print(f"[{idx}/{len(samples)}] Sample {idx}: {status} | Running Accuracy: {accuracy_so_far:.2%}")

                # 可选：显示预测和真实值（用于调试）
                if not is_correct and idx <= 3:  # 只显示前3个错误的例子
                    print(f"  Expected: {ground_truth}")
                    print(f"  Predicted: {predicted}")

            except Exception as e:
                print(f"[{idx}/{len(samples)}] Sample {idx}: Error - {str(e)}")
                total += 1
                continue

        # 计算本轮准确率
        epoch_accuracy = correct / total if total > 0 else 0.0
        epoch_accuracies.append(epoch_accuracy)
        print(f"\nEpoch {epoch} Results: {correct}/{total} = {epoch_accuracy:.2%}")

    # 显示最终结果
    print("\n" + "=" * 50)
    print("=== Final Results ===")
    print(f"Processed {NUM_EPOCHS} epochs with {len(samples)} samples each")

    if NUM_EPOCHS > 1:
        avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
        print(f"Average Accuracy across all epochs: {avg_accuracy:.2%}")
        print(f"Per-epoch accuracies: {[f'{acc:.2%}' for acc in epoch_accuracies]}")
    else:
        print(f"Final Accuracy: {epoch_accuracies[0]:.2%}")


def test_single_sample():
    """
    测试单个样本（用于调试）
    """
    API_KEY = "your_api_key_here"
    BASE_URL = "https://api.openai.com/v1"
    MODEL_NAME = "gpt-4"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 读取第一个样本
    with open("val.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    # 保存ground truth
    ground_truth = sample['test'][0]['output']

    # 准备输入
    sample_input = sample.copy()
    sample_input['test'] = [{'input': sample['test'][0]['input']}]

    # 构造prompt
    messages = construct_prompt(sample_input)

    print("Prompt constructed:")
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]:")
        print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])

    print("\n" + "=" * 50)
    print("Calling API...")

    # 调用API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=2000
    )

    output_text = response.choices[0].message.content
    print("\nModel Output:")
    print(output_text)

    print("\n" + "=" * 50)
    predicted = parse_output(output_text)
    print(f"Parsed Output: {predicted}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Match: {predicted == ground_truth}")


if __name__ == "__main__":
    # 运行主测试
    main()

    # 如果需要测试单个样本，取消注释下面这行
    # test_single_sample()