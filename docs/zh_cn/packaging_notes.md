# X-AnyLabeling Windows-CPU 打包说明（个人笔记）

> 本文件记录打包可执行程序的完整流程和注意事项，防止下次打包出错。

## 1. 打包环境

- 操作系统：Windows
- conda 虚拟环境：`xanylabeling-build`（位于 `D:\anaconda3\envs\xanylabeling-build`）
- 打包工具：PyInstaller（spec 驱动）

## 2. 打包前的检查项

1. **设备配置**：确认 `anylabeling/app_info.py` 中的 `__preferred_device__` 为 `CPU`（打 GPU 版时改为 `GPU`）。
2. **确认 pyinstaller 已安装**：激活环境后执行 `pyinstaller --version`，能输出版本号即可。
3. **依赖完整**：`xanylabeling-build` 环境中需包含项目全部依赖（PyQt6、onnxruntime、torch、torchvision、ultralytics、matplotlib、pandas、openpyxl 等），否则打包出的程序运行时会缺模块。

## 3. 打包命令（PowerShell）

```powershell
# 1. 进入项目根目录
cd d:\desk\anylabeling\X-AnyLabeling

# 2. 激活打包专用环境
conda activate xanylabeling-build

# 3. 设置项目根目录环境变量（spec 文件靠它定位资源，漏掉会出错）
$env:X_ANYLABELING_ROOT = "d:\desk\anylabeling\X-AnyLabeling"

# 4. 执行打包
pyinstaller --noconfirm packaging\pyinstaller\specs\x-anylabeling-win-cpu.spec
```

### 备选：构建脚本（需 Git Bash / WSL）

```bash
bash scripts/build_executable.sh win-cpu
```

脚本会自动设置 `X_ANYLABELING_ROOT` 并调用同一个 spec 文件。

## 4. 打包产物结构

打包完成后输出在 `dist\X-AnyLabeling-CPU\`，目录下只有**一个文件和一个文件夹**：

```
X-AnyLabeling-CPU/
├── X-AnyLabeling-v{版本号}-CPU.exe   ← 主程序（双击启动）
└── _internal/                        ← 全部依赖（Python运行时、DLL、资源等）
```

**关键规则：**
- `.exe` 和 `_internal/` **必须放在一起**，只拷 `.exe` 给别人程序无法运行。
- 版本号从 `anylabeling/app_info.py` 的 `__version__` 读取（当前为 `v4.0.0-beta.10`）。
- 这是 onedir（目录）模式，启动快、稳定，是推荐的分发模式。
- **不要**设置 `X_ANYLABELING_ONEFILE=1` 打单文件 exe：每次启动都要解压到临时目录，慢且易报 `failed to extract entry: LIBBZ2.dll` 错误。

## 5. 分发给别人使用

### 5.1 压缩整个目录

```powershell
cd d:\desk\anylabeling\X-AnyLabeling\dist
Compress-Archive -Path "X-AnyLabeling-CPU" -DestinationPath "X-AnyLabeling-CPU-v4.0.0-beta.10.zip" -Force
```

### 5.2 上传到 GitHub Release（推荐）

GitHub Release 单个文件上限 **2GB**，不占用仓库体积：

1. 打开仓库页面 → **Releases** → **Create a new release**
2. 填写 tag（如 `v4.0.0-beta.10`）、标题、说明
3. 把 zip 拖到 **Attach binaries** 区域
4. **Publish release**

或用 gh CLI：

```powershell
gh release create v4.0.0-beta.10 "X-AnyLabeling-CPU-v4.0.0-beta.10.zip" --title "X-AnyLabeling v4.0.0-beta.10" --notes "CPU 版本发布包"
```

### 5.3 如果 zip 超过 2GB

- **分卷压缩**（需 7-Zip）：
  ```powershell
  & "C:\Program Files\7-Zip\7z.exe" a -v1800m "X-AnyLabeling-CPU.7z" "dist\X-AnyLabeling-CPU"
  ```
  生成 `.7z.001`、`.7z.002` 等分卷，全部上传，用户下载后放在同一目录解压 `.001`。
- **外部网盘**：上传到网盘后在 README 放链接。

## 6. 与 git 提交的关系（重要）

| 内容 | 是否提交到 git | 说明 |
|------|---------------|------|
| 源码（`.py`、`.yaml`、资源图片） | ✅ 是 | `git add -A` + `git commit` + `git push` |
| `dist/` 打包产物 | ❌ 否 | 已在 `.gitignore` 中排除；体积数 GB，且可从源码重新打包 |
| `build/` 构建中间文件 | ❌ 否 | 已在 `.gitignore` 中排除 |
| 模型权重（`.safetensors`、`.pt`） | ⚠️ 走 Git LFS | 已在 `.gitattributes` 配置，注意 LFS 免费额度 1GB |
| 打包好的 zip | ❌ 不进 git | 走 GitHub Release 分发 |

**核心原则**：`git push` 只推源码；二进制产物用 Release 分发。两者是独立的发布渠道。

## 7. 常见错误速查

| 症状 | 原因 | 解决 |
|------|------|------|
| spec 找不到资源/配置文件 | 未设置 `X_ANYLABELING_ROOT` | 打包前 `$env:X_ANYLABELING_ROOT = "d:\desk\anylabeling\X-AnyLabeling"` |
| 打包出的程序运行报 `No module named xxx` | 环境缺依赖，或该模块是函数内懒加载未被静态分析发现 | 在环境里装依赖；或在 spec 的 `hiddenimports` 中显式添加（参考 `pandas`、`openpyxl` 的写法） |
| 程序启动报 `torchvision::nms does not exist` | torchvision 的 `_C_stable.pyd` 未被打包 | spec 已用 `collect_dynamic_libs('torchvision', search_patterns=['*.pyd', '*.dll'])` 处理，确认该段未被删改 |
| 启动报 `DLL load failed ... QtCore` | Conda 的 ICU DLL 覆盖了系统 ICU | spec 已有 `_strip_system_icu_binaries` 处理，确认该段未被删改 |
| `pyinstaller` 命令找不到 | 未激活正确环境 / PATH 问题 | 先 `conda activate xanylabeling-build`；或用绝对路径调用 |
| 打包后 exe 没更新 | 旧的 dist/build 目录干扰 | 使用 `--noconfirm` 参数覆盖，或手动删除 `dist\`、`build\` 后重打 |
