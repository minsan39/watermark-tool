# 水印去除工具 (Watermark Remover)

## 工具描述
这是一个基于AI的水印去除工具，可以自动识别并移除图片中的水印。支持文字水印、图像模板匹配和坐标区域三种模式。

提供两种调用方式：
1. **CLI命令行模式** - 直接执行命令处理图片
2. **API服务模式** - 启动HTTP服务，通过API调用

## 工具路径
```
d:\Python项目\watermark_remover.py
```

---

## 方式一：CLI命令行模式

### 基本命令格式
```bash
python d:\Python项目\watermark_remover.py cli --input <图片路径> --mode <模式> [参数]
```

### 三种模式

#### 1. 文字模式 (text)
通过OCR识别图片中的文字，自动定位并移除匹配的水印文字。

```bash
python d:\Python项目\watermark_remover.py cli -i <图片路径> -m text -t "水印文字"
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `text`
- `-t, --text`: 要移除的水印文字（必需）

#### 2. 图像模板模式 (image)
通过模板图像匹配，找到并移除相似的水印区域。

```bash
python d:\Python项目\watermark_remover.py cli -i <图片路径> -m image -tp <模板图片路径>
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `image`
- `-tp, --template`: 模板图片路径（必需）
- `-th, --threshold`: 匹配阈值 0.1-0.9，默认0.3（可选）

#### 3. 坐标模式 (box)
直接移除指定坐标区域的图像内容。

```bash
python d:\Python项目\watermark_remover.py cli -i <图片路径> -m box -c "x1,y1,x2,y2"
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `box`
- `-c, --coords`: 坐标区域，格式 `x1,y1,x2,y2`（必需）

### CLI其他参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--output` | `-o` | 输出图片路径，不指定则覆盖原文件 |
| `--output-dir` | `-od` | 批量处理时的输出目录 |
| `--batch` | `-b` | 批量处理模式（处理整个文件夹） |
| `--json` | `-j` | 以JSON格式输出结果，方便程序解析 |

---

## 方式二：API服务模式

### 启动API服务
```bash
python d:\Python项目\watermark_remover.py api --port 8080
```

参数：
- `--port, -p`: 服务端口，默认 8080
- `--host`: 服务地址，默认 127.0.0.1

服务启动后可通过 HTTP 请求调用。

### API端点

#### GET / - 服务信息
```
GET http://localhost:8080/
```

响应：
```json
{
  "name": "水印去除工具 API",
  "version": "1.0.0",
  "endpoints": {
    "/remove": "POST - 移除水印",
    "/batch": "POST - 批量处理",
    "/status": "GET - 服务状态"
  }
}
```

#### GET /status - 服务状态
```
GET http://localhost:8080/status
```

响应：
```json
{
  "status": "running",
  "device": "cuda",
  "models_loaded": {
    "lama": false,
    "ocr": false
  }
}
```

#### POST /remove - 移除水印
```
POST http://localhost:8080/remove
Content-Type: application/json
```

请求体：
```json
{
  "image": "图片路径",
  "mode": "text",
  "text": "水印文字",
  "output": "输出路径（可选）"
}
```

**text模式请求示例：**
```json
{
  "image": "D:/photo.jpg",
  "mode": "text",
  "text": "Sample"
}
```

**image模式请求示例：**
```json
{
  "image": "D:/photo.jpg",
  "mode": "image",
  "template": "D:/logo.png",
  "threshold": 0.3
}
```

**box模式请求示例：**
```json
{
  "image": "D:/photo.jpg",
  "mode": "box",
  "coords": [100, 100, 300, 200]
}
```
或
```json
{
  "image": "D:/photo.jpg",
  "mode": "box",
  "coords": "100,100,300,200"
}
```

响应：
```json
{
  "success": true,
  "message": "成功移除水印，找到 1 个匹配区域",
  "output_path": "D:/photo.jpg"
}
```

#### POST /batch - 批量处理
```
POST http://localhost:8080/batch
Content-Type: application/json
```

请求体：
```json
{
  "input": "图片目录或单个文件路径",
  "mode": "text",
  "text": "水印文字",
  "output_dir": "输出目录（可选）"
}
```

响应：
```json
{
  "total": 10,
  "success": 8,
  "failed": 2,
  "results": [
    {
      "success": true,
      "message": "成功移除水印，找到 1 个匹配区域",
      "output_path": "D:/images/photo1.jpg",
      "input_path": "D:/images/photo1.jpg"
    },
    ...
  ]
}
```

---

## 输出格式

### CLI普通输出
```
成功: 成功移除水印，找到 1 个匹配区域
输出: photo.jpg
```

### CLI JSON输出 (使用 --json 参数)
```json
{
  "success": true,
  "message": "成功移除水印，找到 1 个匹配区域",
  "output_path": "photo.jpg"
}
```

---

## 使用示例

### 示例1：CLI移除文字水印
用户说："帮我把 photo.jpg 里的 'Sample' 水印去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py cli -i photo.jpg -m text -t "Sample"
```

### 示例2：CLI批量处理
用户说："把 images 文件夹里所有图片的 '水印' 文字都去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py cli -i ./images/ -m text -t "水印" -b
```

### 示例3：API调用移除水印
先启动服务：
```bash
python d:\Python项目\watermark_remover.py api --port 8080
```

然后发送HTTP请求：
```bash
curl -X POST http://localhost:8080/remove \
  -H "Content-Type: application/json" \
  -d '{"image": "D:/photo.jpg", "mode": "text", "text": "Sample"}'
```

### 示例4：API批量处理
```bash
curl -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '{"input": "D:/images/", "mode": "text", "text": "水印"}'
```

---

## 注意事项

1. 支持的图片格式：jpg, jpeg, png, bmp, webp
2. 首次运行会下载LaMa模型（约200MB）
3. 支持GPU加速（自动检测CUDA）
4. 不指定输出路径时会覆盖原文件
5. 批量处理时可以指定 `--output-dir` 或 `output_dir` 保存到新目录
6. API模式适合需要多次调用的场景，模型只需加载一次

## 调用建议

当用户请求移除水印时：
1. 确认用户提供的图片路径是否存在
2. 根据用户描述选择合适的模式：
   - 用户提到具体文字 → 使用 text 模式
   - 用户提供模板图片 → 使用 image 模式
   - 用户指定坐标区域 → 使用 box 模式
3. 单次处理可用CLI，多次处理建议启动API服务
4. 建议使用 JSON 格式获取结构化输出
5. 处理完成后告知用户结果
