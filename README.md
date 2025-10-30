# TextGrad-qwen

这是一个基于 TextGrad 框架的教程项目，专门针对通义千问（Qwen）模型进行了适配。该项目展示了如何使用 TextGrad 优化各种任务，并通过阿里云 DashScope 平台提供的 Qwen 模型实现具体功能。

## 项目简介

TextGrad 是一个用于自动调整大型语言模型（LLM）输入的库，它可以通过梯度下降的方式优化提示词、代码等可微分变量。本项目通过 LiteLLM 接口集成了阿里云 DashScope 提供的 Qwen 系列模型，包括文本模型和视觉模型。

## 文件说明

- [01-Qwen-Tutorial-Primitives.py](file:///D:/PycharmProject/TextGrad-qwen/01-Qwen-Tutorial-Primitives.py) - TextGrad 基础教程，展示如何修复句子中的错误
- [02-Qwen-Tutorial-Solution-Optimization.py](file:///D:/PycharmProject/TextGrad-qwen/02-Qwen-Tutorial-Solution-Optimization.py) - 解决方案优化示例，改进数学问题的答案
- [03-Qwen-Tutorial-Test-Time-Loss-for-Code.py](file:///D:/PycharmProject/TextGrad-qwen/03-Qwen-Tutorial-Test-Time-Loss-for-Code.py) - 代码优化示例，提升最长递增子序列算法的性能
- [04-Qwen-Tutorial-Prompt-Optimization.py](file:///D:/PycharmProject/TextGrad-qwen/04-Qwen-Tutorial-Prompt-Optimization.py) - 提示词优化示例，在 BBH 数据集上优化系统提示词
- [05-Qwen-Tutorial-MultiModal.py](file:///D:/PycharmProject/TextGrad-qwen/05-Qwen-Tutorial-MultiModal.py) - 多模态示例，使用 Qwen-VL 模型进行图像问答
- [LiteLLMEngine.py](file:///D:/PycharmProject/TextGrad-qwen/LiteLLMEngine.py) - 自定义的 Qwen 引擎，支持文本和视觉模型
- [test-litellm-dashscope.py](file:///D:/PycharmProject/TextGrad-qwen/test-litellm-dashscope.py) - 测试 LiteLLM 与 DashScope 的连接

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
需要在环境变量中设置 `DASHSCOPE_API_KEY`：
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

或者在 Windows 上：
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
```

## 使用方法

每个教程文件都可以单独运行，例如：
```bash
python 01-Qwen-Tutorial-Primitives.py
```

根据使用的模型，可能需要不同的 DashScope API 密钥权限。

## 主要特性

- **TextGrad 集成**：完全兼容 TextGrad 框架的所有核心功能
- **Qwen 模型支持**：支持 Qwen 系列文本模型（如 qwen-max, qwen-plus）
- **多模态支持**：支持 Qwen-VL 系列视觉语言模型
- **LiteLLM 后端**：通过 LiteLLM 统一接口访问 DashScope 服务

## 自定义引擎

[LiteLLMEngine.py](file:///D:/PycharmProject/TextGrad-qwen/LiteLLMEngine.py) 包含两个自定义引擎类：

1. `QwenEngine` - 用于文本模型
2. `QwenVisionEngine` - 用于视觉模型

这些引擎实现了 TextGrad 所需的接口，可以直接在 TextGrad 工作流中使用。

## 注意事项

- 需要有可用的网络连接来访问 DashScope API
- 使用前请确保设置了正确的 `DASHSCOPE_API_KEY`
- 根据使用的模型不同，可能会产生相应的费用