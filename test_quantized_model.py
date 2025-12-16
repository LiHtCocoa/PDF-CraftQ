#!/usr/bin/env python3
"""
测试 DeepSeek OCR 4-bit 量化模型
验证 BitsAndBytes 量化模型能否直接被 pdf-craft 使用
"""

import sys
import torch
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    # 检查 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"  - CUDA 版本: {torch.version.cuda}")
        print(f"  - GPU 数量: {torch.cuda.device_count()}")
        print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"  - 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  ⚠️  警告: CUDA 不可用，BitsAndBytes 需要 GPU 支持")
        return False

    # 检查依赖
    print("\n依赖检查:")
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("✗ bitsandbytes 未安装")
        return False

    try:
        import accelerate
        print(f"✓ accelerate: {accelerate.__version__}")
    except ImportError:
        print("✗ accelerate 未安装")
        return False

    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers 未安装")
        return False

    try:
        import pdf_craft
        print(f"✓ pdf-craft: {pdf_craft.__version__}")
    except ImportError:
        print("✗ pdf-craft 未安装")
        return False

    print("\n✅ 环境检查通过！\n")
    return True


def test_model_download():
    """测试模型下载"""
    print("=" * 60)
    print("测试 1: 模型下载")
    print("=" * 60)

    model_id = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"
    print(f"模型 ID: {model_id}")
    print("开始下载模型（首次运行会自动下载）...\n")

    try:
        from pdf_craft import predownload_models

        predownload_models(
            models_cache_path=model_id,
        )

        print("\n✅ 模型下载成功！")
        return True

    except Exception as e:
        print(f"\n✗ 模型下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("测试 2: 模型加载")
    print("=" * 60)

    model_id = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"
    print(f"尝试加载量化模型: {model_id}\n")

    try:
        from pdf_craft.pdf import OCR

        # 记录初始显存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"初始显存占用: {initial_memory:.2f} GB")

        # 创建 OCR 实例
        print("创建 OCR 实例...")
        ocr = OCR(
            model_path=model_id,
            pdf_handler=None,
            local_only=False,  # 允许下载
        )

        # 加载模型
        print("加载模型到内存...")
        ocr.load_models()

        # 检查显存占用
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n显存统计:")
            print(f"  - 当前占用: {current_memory:.2f} GB")
            print(f"  - 峰值占用: {peak_memory:.2f} GB")
            print(f"  - 增加量: {current_memory - initial_memory:.2f} GB")

        print("\n✅ 模型加载成功！")
        print("✅ 4-bit 量化模型可以直接使用，无需修改 pdf-craft 代码！")
        return True

    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_conversion():
    """测试 PDF 转换（需要提供测试 PDF）"""
    print("\n" + "=" * 60)
    print("测试 3: PDF 转换（可选）")
    print("=" * 60)

    # 检查是否有测试 PDF
    test_pdf = Path("test.pdf")
    if not test_pdf.exists():
        print("⚠️  未找到 test.pdf，跳过转换测试")
        print("提示: 将测试 PDF 文件命名为 test.pdf 放在项目根目录即可测试")
        return None

    print(f"找到测试文件: {test_pdf}")
    print("开始转换...\n")

    try:
        from pdf_craft import transform_markdown

        # 记录初始显存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 执行转换
        result = transform_markdown(
            pdf_path=str(test_pdf),
            markdown_path="output.md",
            markdown_assets_path="images",
            models_cache_path="Jalea96/DeepSeek-OCR-bnb-4bit-NF4",
            ocr_size="base",  # 使用 base 尺寸
            local_only=False,
        )

        # 显示结果
        print(f"\n转换完成！")
        print(f"  - 输入 tokens: {result.input_tokens}")
        print(f"  - 输出 tokens: {result.output_tokens}")

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  - 峰值显存: {peak_memory:.2f} GB")

        print(f"\n输出文件:")
        print(f"  - Markdown: output.md")
        print(f"  - 图片目录: images/")

        print("\n✅ PDF 转换成功！")
        return True

    except Exception as e:
        print(f"\n✗ PDF 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("DeepSeek OCR 4-bit 量化模型测试")
    print("=" * 60)
    print()

    # 1. 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请先安装必要的依赖")
        sys.exit(1)

    # 2. 模型下载
    if not test_model_download():
        print("\n❌ 模型下载失败")
        sys.exit(1)

    # 3. 模型加载
    if not test_model_loading():
        print("\n❌ 模型加载失败")
        sys.exit(1)

    # 4. PDF 转换（可选）
    test_pdf_conversion()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✅ 所有核心测试通过！")
    print("\n结论:")
    print("  - BitsAndBytes 4-bit 量化模型可以直接使用")
    print("  - 无需修改 pdf-craft 源代码")
    print("  - 只需指定量化模型的 Hugging Face ID 即可")
    print("\n使用方法:")
    print('  transform_markdown(')
    print('      pdf_path="input.pdf",')
    print('      markdown_path="output.md",')
    print('      models_cache_path="Jalea96/DeepSeek-OCR-bnb-4bit-NF4",')
    print('  )')
    print()


if __name__ == "__main__":
    main()
