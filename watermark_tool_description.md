# 水印去除工具 (Watermark Remover)

## 工具描述
这是一个基于AI的水印去除工具，可以自动识别并移除图片中的水印。支持文字水印、图像模板匹配和坐标区域三种模式。

## 工具路径
```
d:\Python项目\watermark_remover.py
```

## 使用方式

### 基本命令格式
```bash
python d:\Python项目\watermark_remover.py --input <图片路径> --mode <模式> [参数]
```

### 三种模式

#### 1. 文字模式 (text)
通过OCR识别图片中的文字，自动定位并移除匹配的水印文字。

```bash
python d:\Python项目\watermark_remover.py -i <图片路径> -m text -t "水印文字"
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `text`
- `-t, --text`: 要移除的水印文字（必需）

#### 2. 图像模板模式 (image)
通过模板图像匹配，找到并移除相似的水印区域。

```bash
python d:\Python项目\watermark_remover.py -i <图片路径> -m image -tp <模板图片路径>
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `image`
- `-tp, --template`: 模板图片路径（必需）
- `-th, --threshold`: 匹配阈值 0.1-0.9，默认0.3（可选）

#### 3. 坐标模式 (box)
直接移除指定坐标区域的图像内容。

```bash
python d:\Python项目\watermark_remover.py -i <图片路径> -m box -c "x1,y1,x2,y2"
```

参数：
- `-i, --input`: 输入图片路径（必需）
- `-m, --mode`: 设为 `box`
- `-c, --coords`: 坐标区域，格式 `x1,y1,x2,y2`（必需）

### 其他参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--output` | `-o` | 输出图片路径，不指定则覆盖原文件 |
| `--output-dir` | `-od` | 批量处理时的输出目录 |
| `--batch` | `-b` | 批量处理模式（处理整个文件夹） |
| `--json` | `-j` | 以JSON格式输出结果，方便程序解析 |

### 输出格式

#### 普通输出
```
成功: 成功移除水印，找到 1 个匹配区域
输出: photo.jpg
```

#### JSON输出 (使用 --json 参数)
```json
{
  "success": true,
  "message": "成功移除水印，找到 1 个匹配区域",
  "output_path": "photo.jpg"
}
```

## 使用示例

### 示例1：移除文字水印
用户说："帮我把 photo.jpg 里的 'Sample' 水印去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py -i photo.jpg -m text -t "Sample"
```

### 示例2：批量处理
用户说："把 images 文件夹里所有图片的 '水印' 文字都去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py -i ./images/ -m text -t "水印" -b
```

### 示例3：使用模板移除Logo
用户说："用 logo.png 作为模板，把 photo.jpg 里的 logo 去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py -i photo.jpg -m image -tp logo.png
```

### 示例4：移除指定区域
用户说："把 photo.jpg 左上角 100,100 到 300,200 区域的内容去掉"

执行命令：
```bash
python d:\Python项目\watermark_remover.py -i photo.jpg -m box -c "100,100,300,200"
```

## 注意事项

1. 支持的图片格式：jpg, jpeg, png, bmp, webp
2. 首次运行会下载LaMa模型（约200MB）
3. 支持GPU加速（自动检测CUDA）
4. 不指定输出路径时会覆盖原文件
5. 批量处理时可以指定 `--output-dir` 保存到新目录

## 调用建议

当用户请求移除水印时：
1. 确认用户提供的图片路径是否存在
2. 根据用户描述选择合适的模式：
   - 用户提到具体文字 → 使用 text 模式
   - 用户提供模板图片 → 使用 image 模式
   - 用户指定坐标区域 → 使用 box 模式
3. 建议使用 `--json` 参数获取结构化输出
4. 处理完成后告知用户结果
