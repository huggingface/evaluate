<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/evaluate/media/resolve/main/evaluate-banner.png" width="400"/>
    <br>
</p>

<p align="center">
    <a href="https://github.com/huggingface/evaluate/actions/workflows/ci.yml?query=branch%3Amain">
        <img alt="Build" src="https://github.com/huggingface/evaluate/actions/workflows/ci.yml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/huggingface/evaluate/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/evaluate.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/evaluate/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/evaluate/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/evaluate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/evaluate.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

<p align="center">
    <a href="README.md">English</a> · <b>简体中文</b>
</p>

> **提示：** 对于更新的评估方法（例如评估大语言模型 LLM），我们推荐使用我们更新且更积极维护的库 [LightEval](https://github.com/huggingface/lighteval)。

🤗 Evaluate 是一个旨在让模型评估、模型对比以及性能汇报变得更加简单和标准化的库。

它目前包含：

- **涵盖数十种常用评估指标的实现**：现有指标涵盖从自然语言处理（NLP）到计算机视觉（CV）的多种任务，并包含针对特定数据集的评测指标。通过类似于 `accuracy = load("accuracy")` 这样简洁的命令，即可在任意主流框架（Numpy/Pandas/PyTorch/TensorFlow/JAX）中直接使用这些指标评估机器学习模型。
- **模型对比（Comparisons）与测量工具（Measurements）**：对比工具用于衡量不同模型之间的差异，而测量工具则是用于分析与评估数据集特征的实用工具。
- **便捷地向 🤗 Hub 添加全新评估模块**：您可以使用 `evaluate-cli create [metric name]` 创建新的评估模块并将其推送到 🤗 Hub 上的专属 Space，从而在相同参考标签与预测结果下轻松对比不同指标及其输出。

[🎓 **官方文档**](https://huggingface.co/docs/evaluate/)

🔎 **在 Hub 上查找[评测指标 (Metric)](https://huggingface.co/evaluate-metric)、[模型对比 (Comparison)](https://huggingface.co/evaluate-comparison)与[数据集测量 (Measurement)](https://huggingface.co/evaluate-measurement)**

[🌟 **添加全新评估模块**](https://huggingface.co/docs/evaluate/)

🤗 Evaluate 还拥有许多实用的特性：

- **类型检查（Type checking）**：自动验证输入数据格式，确保每个评测指标传入的输入结构正确无误。
- **指标卡片（Metric cards）**：每个指标均配有专门的卡片文档，详细说明其数值含义、取值范围、局限性，并提供使用示例与应用价值说明。
- **社区驱动的指标生态（Community metrics）**：各项指标均托管在 Hugging Face Hub 上，您可以轻松为自己的项目创建专属指标，或与社区开发者共同协作。

# 安装

## 使用 pip 安装

🤗 Evaluate 已发布至 PyPI，建议在虚拟环境（例如 venv 或 conda）中安装：

```bash
pip install evaluate
```

# 使用方法

🤗 Evaluate 的核心方法包括：

- `evaluate.list_evaluation_modules()`：列出所有可用的指标、对比工具与测量工具
- `evaluate.load(module_name, **kwargs)`：实例化评估模块
- `results = module.compute(*kwargs)`：计算并返回评估模块的结果

# 添加全新评估模块

首先运行以下命令安装创建新指标所需的依赖项：
```bash
pip install evaluate[template]
```
随后执行以下命令，该命令会为您的指标创建专属文件夹并展示后续步骤指引：
```bash
evaluate-cli create "Awesome Metric"
```
详细操作说明请参阅官方文档中的[分步指南](https://huggingface.co/docs/evaluate/creating_and_sharing)。

## 致谢

感谢 [@marella](https://github.com/marella) 允许我们使用此前由其[开源库](https://github.com/marella/evaluate)占用的 PyPI `evaluate` 命名空间。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
