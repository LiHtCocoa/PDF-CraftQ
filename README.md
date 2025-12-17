# PDF-CraftQ

使用[PDF-Craft](https://github.com/oomol-lab/pdf-craft)，在运行时动态注入DeepSeek OCR量化模型，使部署显存占用降低能够在中低端消费级硬件上运行。

## 依赖安装

### 1. 系统依赖

#### Poppler (PDF 渲染必需)

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
1. 从 [poppler releases](https://github.com/osber/poppler-windows/releases) 下载
2. 解压到任意目录（如 `C:\poppler`）
3. 将 `bin` 目录添加到系统 PATH 环境变量

### 2. Python 依赖

推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
cd dsocr-quant-demo
uv sync
```

或使用 pip：

```bash
pip install pdf-craft torch torchvision bitsandbytes accelerate transformers
```

### 3. 硬件要求

- NVIDIA GPU (CUDA 支持)
- 至少 4GB 显存
- 推荐 8GB+ 显存以获得更好性能

## Quick Start

### 方式一：命令行工具 (推荐)

安装后可直接使用 `pdf-craftq` 命令：

```bash
# 激活虚拟环境
source .venv/bin/activate

# PDF 转 Markdown
pdf-craftq input.pdf -o output.md

# PDF 转 EPUB
pdf-craftq input.pdf -o output.epub

# 指定 OCR 模型大小 (tiny/small/base/large/gundam)
pdf-craftq input.pdf -o output.md --ocr-size base

# 详细输出
pdf-craftq input.pdf -o output.md -v
```

更多选项：
```bash
pdf-craftq --help
```

### 方式二：Python API

```python
# 重要：必须在导入 pdf_craft 之前应用 patch
from quantized_model import apply_quantized_model_patch
apply_quantized_model_patch()

# 然后正常使用 pdf_craft
from pdf_craft import transform_markdown, transform_epub

# 转换为 Markdown
result = transform_markdown(
    pdf_path="input.pdf",
    markdown_path="output.md",
    markdown_assets_path="images",  # 图片保存目录
    ocr_size="base",
)

print(f"输入 tokens: {result.input_tokens}")
print(f"输出 tokens: {result.output_tokens}")

# 转换为 EPUB
result = transform_epub(
    pdf_path="input.pdf",
    epub_path="output.epub",
    ocr_size="base",
)
```

### 方式三：运行测试脚本

```bash
source .venv/bin/activate
python test_quantized_model.py
```

测试脚本会依次执行：
1. 环境检查 (CUDA、依赖版本)
2. 模型下载 (自动从 HuggingFace 下载量化模型)
3. 模型加载 (验证 4-bit 加载，显示显存占用)
4. PDF 转换 (可选，需要 `test.pdf`)

## 工作原理

通过 monkey-patch 方式动态替换 `doc_page_extractor` 中的原始模型类，使其加载预量化的 4-bit 模型 (`Jalea96/DeepSeek-OCR-bnb-4bit-NF4`) 而非官方原始模型。

## 项目结构

```
dsocr-quant-demo/
├── cli.py                  # 命令行工具入口
├── quantized_model.py      # 量化模型实现和 monkey-patch
├── test_quantized_model.py # 测试脚本
├── pyproject.toml          # 项目配置和依赖
└── README.md
```

## OCR 模型大小选项

| 选项 | 图像尺寸 | 适用场景 |
|------|----------|----------|
| tiny | 512x512 | 快速预览，低显存 |
| small | 640x640 | 一般文档 |
| base | 1024x1024 | 推荐，平衡质量和速度 |
| large | 1280x1280 | 高质量需求 |
| gundam | 1024x640 | 特殊裁剪模式 |

## 常见问题

### CUDA 不可用
确保已安装 NVIDIA 驱动和 CUDA toolkit，并且 PyTorch 是 CUDA 版本：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 显存不足
尝试使用更小的 OCR 模型：
```bash
pdf-craftq input.pdf -o output.md --ocr-size tiny
```

### Poppler 未找到
确保 `pdftoppm` 命令可用：
```bash
pdftoppm -v
```
