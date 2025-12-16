1. 快速开始

# 安装依赖
cd /mnt/d/CodeBase/llm_playground/dsocr-quant-demo
uv sync
# 安装poppler
(待补充)

# 运行测试
python test_quantized_model.py

# 工作原理

动态注入DS OCR类实现量化模型替换

---
🎯 关键文件说明

test_quantized_model.py - 完整测试脚本

包含 4 个测试：
1. 环境检查：CUDA、依赖版本
2. 模型下载：自动从 HF 下载
3. 模型加载：验证 4-bit 加载，显示显存占用
4. PDF 转换：可选，需要 test.pdf

---
🚀 使用步骤

1. 同步依赖

cd /mnt/d/CodeBase/llm_playground/dsocr-quant-demo
uv sync
安装poppler

2. 运行测试（会自动下载模型）

source .venv/bin/activate
python test_quantized_model.py

3. 测试 PDF 转换（可选）

# 放置测试 PDF
cp /path/to/your.pdf test.pdf
