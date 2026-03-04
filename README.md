# 🎨 Watermark Tool - AI驱动的水印去除工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI Ready](https://img.shields.io/badge/AI-Ready-purple.svg)](https://github.com/minsan39/watermark-tool)

一款基于 AI 的智能水印去除工具，**专为 AI 助手调用设计**，同时提供友好的图形界面供人类用户使用。

## ✨ 核心特性

- 🤖 **AI 优先设计** - 暴露标准化 CLI 和 API 接口，供 AI 助手直接调用
- 🔍 **自动水印检测** - 内置 OCR + 视觉模型，自动识别水印位置和内容
- 🚀 **一键自动处理** - auto 模式：检测 → 定位 → 去除，全程自动化
- 🖼️ **多种去除模式** - 支持文字、图像模板、坐标区域三种模式
- 📦 **批量处理** - 支持文件夹批量处理
- 💻 **友好 GUI** - 图形界面，拖拽操作，适合非技术用户

## 🎯 适用场景

| 用户类型 | 使用方式 | 适用场景 |
|---------|---------|---------|
| **普通用户** | GUI 图形界面 | 单张/少量图片处理，可视化操作 |
| **开发者** | CLI 命令行 | 脚本集成、自动化任务 |
| **AI 助手** | CLI / API | Claude、Cursor、Trae、OpenCode 等 AI 直接调用 |

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision opencv-python pillow numpy
pip install simple-lama-inpainting rapidocr-onnxruntime
pip install transformers flask tkinterdnd2
```

### 运行方式

```bash
# GUI 图形界面（默认）
python watermark_remover.py

# CLI 命令行
python watermark_remover.py cli --input photo.jpg --mode text --text "水印文字"

# API 服务
python watermark_remover.py api --port 8080
```

---

## 🤖 AI 助手调用指南

本工具**专为 AI 助手设计**，提供标准化接口，支持以下 AI 平台直接调用：

| AI 平台 | 调用方式 | 说明 |
|--------|---------|------|
| **Claude (Anthropic)** | CLI / API | 通过 Bash 工具执行命令 |
| **Cursor** | CLI / API | 终端命令执行 |
| **Trae** | CLI / API | 终端命令执行 |
| **OpenCode** | CLI / API | 终端命令执行 |
| **GitHub Copilot** | CLI | 终端命令执行 |
| **其他 AI 助手** | CLI / API | 只需支持终端执行或 HTTP 请求 |

### AI 调用方式一：auto 自动模式（推荐）

**一条命令完成检测 + 去除**，无需两步操作：

```bash
python watermark_remover.py auto -i <图片路径> --json
```

**返回结果：**
```json
{
  "success": true,
  "mode_used": "text",
  "message": "自动去除成功 (模式: text)",
  "detect_result": {
    "success": true,
    "watermark_info": {
      "type": "text",
      "content": "豆包AI生成",
      "location": "bottom-right",
      "position": [1483, 2264, 1723, 2315],
      "confidence": 0.85
    }
  },
  "remove_result": {
    "success": true,
    "message": "成功移除水印",
    "output_path": "photo.jpg"
  }
}
```

### AI 调用方式二：detect + remove 分步模式

**步骤 1：检测水印**
```bash
python watermark_remover.py detect -i photo.jpg --json
```

**返回：**
```json
{
  "success": true,
  "caption": "图片底部有灰色文字'豆包AI生成'",
  "watermark_info": {
    "type": "text",
    "content": "豆包AI生成",
    "location": "bottom-right",
    "position": [1483, 2264, 1723, 2315],
    "confidence": 0.85
  }
}
```

**步骤 2：根据检测结果移除**
```bash
# 文字水印 - 使用 content
python watermark_remover.py cli -i photo.jpg -m text -t "豆包AI生成"

# 图像水印 - 使用 position
python watermark_remover.py cli -i photo.jpg -m box -c "1483,2264,1723,2315"
```

### AI 调用方式三：API 服务模式

适合需要多次调用的场景，模型只需加载一次：

```bash
# 启动服务
python watermark_remover.py api --port 8080
```

**API 端点：**

| 端点 | 方法 | 说明 |
|-----|------|------|
| `/` | GET | 服务信息 |
| `/status` | GET | 服务状态 |
| `/detect` | POST | 检测水印 |
| `/remove` | POST | 移除水印 |
| `/auto` | POST | 自动检测并移除 |
| `/batch` | POST | 批量处理 |

**调用示例：**

```bash
# 自动检测并移除
curl -X POST http://localhost:8080/auto \
  -H "Content-Type: application/json" \
  -d '{"image": "D:/photo.jpg"}'

# 检测水印
curl -X POST http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "D:/photo.jpg"}'

# 移除水印
curl -X POST http://localhost:8080/remove \
  -H "Content-Type: application/json" \
  -d '{"image": "D:/photo.jpg", "mode": "text", "text": "水印文字"}'
```

---

## 📋 CLI 命令参考

### auto - 自动模式（推荐 AI 使用）

```bash
python watermark_remover.py auto -i <图片路径> [-o 输出路径] [--json]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `--input` | `-i` | 输入图片路径（必需） |
| `--output` | `-o` | 输出路径（可选，默认覆盖原文件） |
| `--json` | `-j` | JSON 格式输出 |
| `--no-caption` | | 不生成图片描述（更快） |

### detect - 检测水印

```bash
python watermark_remover.py detect -i <图片路径> [--json]
```

### cli - 命令行处理

```bash
python watermark_remover.py cli -i <图片路径> -m <模式> [参数]
```

**三种模式：**

| 模式 | 说明 | 必需参数 |
|------|------|---------|
| `text` | 文字水印 | `-t "水印文字"` |
| `image` | 图像模板 | `-tp 模板图片路径` |
| `box` | 坐标区域 | `-c "x1,y1,x2,y2"` |

**通用参数：**

| 参数 | 简写 | 说明 |
|------|------|------|
| `--output` | `-o` | 输出路径 |
| `--batch` | `-b` | 批量处理 |
| `--output-dir` | `-od` | 批量输出目录 |
| `--json` | `-j` | JSON 格式输出 |

### api - API 服务

```bash
python watermark_remover.py api [--port 8080] [--host 127.0.0.1]
```

---

## 👤 人类用户指南

### GUI 图形界面

直接运行程序即可启动图形界面：

```bash
python watermark_remover.py
```

**功能说明：**

1. **打开图片** - 点击"打开"按钮或拖拽图片到窗口
2. **选择模式**：
   - **框选模式** - 鼠标拖拽选择水印区域
   - **文字模式** - 输入水印文字，自动 OCR 定位
   - **图像模式** - 加载水印模板，自动匹配
   - **自动模式** - 一键自动检测并移除
3. **移除水印** - 点击"移除"按钮
4. **保存图片** - 点击"保存"按钮

**批量处理：**
1. 打开多张图片（支持多选）
2. 选择模式（文字/图像/自动）
3. 点击"批量移除"
4. 点击"保存所有"

---

## 📖 使用示例

### 示例 1：AI 助手自动处理

**用户对 AI 说：** "帮我把 photo.jpg 里的水印去掉"

**AI 执行：**
```bash
python watermark_remover.py auto -i photo.jpg --json
```

**AI 回复：** "已成功移除水印！检测到文字水印'豆包AI生成'，已自动处理完成。"

### 示例 2：指定文字水印

**用户对 AI 说：** "帮我把图片里的 'Sample' 水印去掉"

**AI 执行：**
```bash
python watermark_remover.py cli -i photo.jpg -m text -t "Sample"
```

### 示例 3：批量处理

**用户对 AI 说：** "把 images 文件夹里所有图片的水印都去掉"

**AI 执行：**
```bash
python watermark_remover.py cli -i ./images/ -m auto -b --json
```

### 示例 4：API 模式

**启动服务：**
```bash
python watermark_remover.py api --port 8080
```

**AI 调用：**
```bash
curl -X POST http://localhost:8080/auto \
  -H "Content-Type: application/json" \
  -d '{"image": "D:/photo.jpg"}'
```

---

## 🔧 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Watermark Tool                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │   GUI   │  │   CLI   │  │   API   │  │  Auto   │    │
│  │ 图形界面 │  │ 命令行  │  │ HTTP服务│  │ 自动模式 │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       └────────────┴────────────┴────────────┘          │
│                         │                                │
│  ┌──────────────────────┴──────────────────────┐       │
│  │              WatermarkDetector               │       │
│  │  ┌─────────────┐  ┌─────────────────────┐   │       │
│  │  │  RapidOCR   │  │  Florence-2 (可选)  │   │       │
│  │  │  文字检测   │  │  图像理解/描述      │   │       │
│  │  └─────────────┘  └─────────────────────┘   │       │
│  └─────────────────────────────────────────────┘       │
│                         │                                │
│  ┌──────────────────────┴──────────────────────┐       │
│  │              WatermarkRemover                │       │
│  │  ┌─────────────────────────────────────┐    │       │
│  │  │        LaMa (图像修复模型)           │    │       │
│  │  │        SimpleLama Inpainting         │    │       │
│  │  └─────────────────────────────────────┘    │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 模型说明

首次运行会自动下载以下模型：

| 模型 | 大小 | 用途 |
|------|------|------|
| **LaMa** | ~200MB | 图像修复/水印去除 |
| **RapidOCR** | ~50MB | 文字检测识别 |
| **Florence-2** | ~900MB | 图像理解/描述（可选） |

**注意：** Florence-2 模型用于生成图片描述，如果不使用 `--caption` 功能可以不加载。

---

## ⚙️ 配置选项

### 环境变量

```bash
# 使用 HuggingFace 镜像（国内用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 禁用 SSL 验证（网络问题时使用）
export HF_HUB_DISABLE_SSL_VERIFY=1
```

### GPU 加速

工具自动检测 CUDA，如有 GPU 会自动使用。检查方法：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 🤝 支持的 AI 平台

本工具的 CLI 和 API 接口设计遵循以下原则，确保与主流 AI 助手兼容：

1. **标准化输出** - JSON 格式，易于解析
2. **清晰的错误信息** - 包含详细的错误描述
3. **幂等操作** - 重复调用不会产生副作用
4. **路径灵活性** - 支持绝对路径和相对路径

**已验证兼容的 AI 平台：**

- ✅ Claude (Anthropic) - Bash 工具
- ✅ Cursor - 终端执行
- ✅ Trae - 终端执行
- ✅ OpenCode - 终端执行
- ✅ GitHub Copilot - 终端执行
- ✅ ChatGPT (Code Interpreter) - 脚本执行
- ✅ 任何支持终端命令执行的 AI 助手

---

## 📝 更新日志

### v1.2.0
- ✨ 新增 auto 自动模式：一键检测并移除
- ✨ GUI 新增"自动"模式选项
- 🐛 修复 Florence-2 模型兼容性问题
- 🎯 优化水印选择逻辑（关键词加分、年份降权）

### v1.1.0
- ✨ 新增水印自动检测功能
- ✨ 新增 CLI detect 子命令
- ✨ 新增 API /detect 端点

### v1.0.0
- 🎉 初始版本
- ✨ 支持文字、图像模板、坐标三种去除模式
- ✨ GUI 图形界面
- ✨ CLI 命令行
- ✨ API 服务

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [LaMa](https://github.com/advimman/lama) - 图像修复模型
- [SimpleLama](https://github.com/enesmsahin/simple-lama-inpainting) - LaMa 的简化封装
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - 高性能 OCR 引擎
- [Florence-2](https://github.com/microsoft/Florence-2) - 微软视觉模型

---

<p align="center">
  Made with ❤️ for AI assistants and human users
</p>
