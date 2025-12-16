#!/usr/bin/env python3
"""
简单示例：使用 4-bit 量化模型进行 PDF 转换
"""

from pdf_craft import transform_markdown
from pathlib import Path


def main():
    # 检查测试文件
    pdf_path = Path("test.pdf")
    if not pdf_path.exists():
        print("❌ 未找到 test.pdf")
        print("请将测试 PDF 文件放在项目根目录并命名为 test.pdf")
        return

    print("🚀 开始转换 PDF...")
    print(f"输入文件: {pdf_path}")
    print("使用模型: Jalea96/DeepSeek-OCR-bnb-4bit-NF4 (4-bit 量化)")
    print()

    # 使用量化模型进行转换
    result = transform_markdown(
        pdf_path=str(pdf_path),
        markdown_path="output.md",
        markdown_assets_path="images",
        models_cache_path="Jalea96/DeepSeek-OCR-bnb-4bit-NF4",  # 4-bit 量化模型
        ocr_size="base",
        includes_footnotes=True,
    )

    print("✅ 转换完成！")
    print(f"输入 tokens: {result.input_tokens}")
    print(f"输出 tokens: {result.output_tokens}")
    print()
    print("输出文件:")
    print("  - output.md")
    print("  - images/")


if __name__ == "__main__":
    main()
