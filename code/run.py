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
    API_KEY = "sk-d2171e95a0364e86a5a9148aeab43670"  # 替换为你的API密钥
    BASE_URL = "https://api.deepseek.com"  # API endpoint
    MODEL_NAME = "deepseek-chat"
    TEMPERATURE = 1.0  # 温度参数
    NUM_EPOCHS = 1  # 评估轮数
    MAX_ERRORS_TO_SHOW = 3  # 最多显示几个错误样本的详细信息

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
        errors_shown = 0  # 记录已经显示的错误数量
        failed_samples = 0  # 记录失败的样本数

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

                # 计算当前运行准确率（基于成功完成的样本）
                accuracy_so_far = correct / total
                status = "✓" if is_correct else "✗"
                print(f"[{idx}/{len(samples)}] Sample {idx}: {status} | Running Accuracy: {accuracy_so_far:.2%}")

                # 只显示前MAX_ERRORS_TO_SHOW个错误的详细信息
                if not is_correct and errors_shown < MAX_ERRORS_TO_SHOW:
                    print(f"  Expected: {ground_truth}")
                    print(f"  Predicted: {predicted}")
                    errors_shown += 1

            except Exception as e:
                # 改进的错误处理
                error_msg = str(e)
                failed_samples += 1

                # 识别不同类型的错误
                if "timeout" in error_msg.lower():
                    print(f"[{idx}/{len(samples)}] Sample {idx}: Error - Request timed out.")
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    print(f"[{idx}/{len(samples)}] Sample {idx}: Error - Network connection issue.")
                elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
                    print(f"[{idx}/{len(samples)}] Sample {idx}: Error - Rate limit exceeded.")
                elif "api" in error_msg.lower() and "key" in error_msg.lower():
                    print(f"[{idx}/{len(samples)}] Sample {idx}: Error - API key issue.")
                else:
                    # 对于其他错误，显示简短的错误信息
                    short_error = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
                    print(f"[{idx}/{len(samples)}] Sample {idx}: Error - {short_error}")

                # 网络错误不计入total，这样准确率只基于成功完成的样本
                continue

        # 计算本轮准确率
        if total > 0:
            epoch_accuracy = correct / total
            epoch_accuracies.append(epoch_accuracy)
            print(f"\nEpoch {epoch} Results:")
            print(f"  Successfully processed: {total}/{len(samples)} samples")
            print(f"  Failed samples: {failed_samples}")
            print(f"  Correct predictions: {correct}/{total}")
            print(f"  Accuracy (on completed samples): {epoch_accuracy:.2%}")
        else:
            print(f"\nEpoch {epoch}: No samples completed successfully")

    # 显示最终结果
    print("\n" + "=" * 50)
    print("=== Final Results ===")
    print(f"Total samples in dataset: {len(samples)}")

    if epoch_accuracies:
        if NUM_EPOCHS > 1:
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            print(f"Average Accuracy across all epochs: {avg_accuracy:.2%}")
            print(f"Per-epoch accuracies: {[f'{acc:.2%}' for acc in epoch_accuracies]}")
        else:
            print(f"Final Accuracy: {epoch_accuracies[0]:.2%}")
    else:
        print("No successful completions")


def test_single_sample(sample_index=0):
    """
    测试单个样本（用于调试）

    参数:
    sample_index: 要测试的样本索引（从0开始）
    """
    API_KEY = "sk-32537d6922344b1faa9f853e971b23d4"
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 读取指定样本
    with open("val.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sample_index >= len(lines):
            print(f"Error: Sample index {sample_index} out of range (max: {len(lines) - 1})")
            return
        sample = json.loads(lines[sample_index])

    print(f"Testing Sample {sample_index + 1}")
    print("=" * 50)

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
        content = msg['content']
        if len(content) > 800:
            print(content[:800] + "\n... [truncated]")
        else:
            print(content)

    print("\n" + "=" * 50)
    print("Calling API...")

    try:
        # 调用API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=1.0,
            max_tokens=8000
        )

        output_text = response.choices[0].message.content
        print("\nModel Output (first 1500 chars):")
        print(output_text[:1500] + "..." if len(output_text) > 1500 else output_text)

        print("\n" + "=" * 50)
        predicted = parse_output(output_text)
        print(f"Parsed Output: {predicted}")
        print(f"Ground Truth: {ground_truth}")

        is_match = predicted == ground_truth
        print(f"Exact Match: {'✓ YES' if is_match else '✗ NO'}")

        if not is_match:
            print("\nDetailed Comparison:")
            print(
                f"Predicted dimensions: {len(predicted)}x{len(predicted[0]) if predicted and len(predicted) > 0 else 0}")
            print(
                f"Expected dimensions: {len(ground_truth)}x{len(ground_truth[0]) if ground_truth and len(ground_truth) > 0 else 0}")

            # 显示差异位置
            if len(predicted) == len(ground_truth) and all(len(pred_row) == len(gt_row)
                                                           for pred_row, gt_row in zip(predicted, ground_truth)):
                print("\nDifferences at positions:")
                for i in range(len(predicted)):
                    for j in range(len(predicted[i])):
                        if predicted[i][j] != ground_truth[i][j]:
                            print(f"  Position [{i}][{j}]: predicted={predicted[i][j]}, expected={ground_truth[i][j]}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # 运行主测试
    main()

    # 如果需要测试特定样本，使用下面的代码
    # test_single_sample(1)  # 测试第2个样本（索引从0开始）