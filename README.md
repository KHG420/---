# 肺炎 X 光辅助诊断系统 — 启动指南

基于 ResNet50 + CBAM 注意力机制的肺炎 X 光辅助诊断系统（4 分类：新冠肺炎、正常、肺部阴影、病毒性肺炎）。

---

## 环境要求

| 组件   | 版本                  |
|--------|-----------------------|
| Python | >= 3.12（当前使用 3.14.4） |
| Node.js | >= 18（当前使用 22.22） |
| npm    | >= 9（当前使用 9.2） |
| 端口   | 5002（后端 Flask）、5173（前端 Vite） |

---

## 后端启动

```bash
# 1. 进入项目目录
cd /home/aq/深度学习课设/pneumonia-diagnosis

# 2. 激活虚拟环境
source backend/venv/bin/activate

# 3. （首次）安装依赖
pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 启动 Flask API 服务
cd backend
python app.py --port 5002
```

看到以下输出即启动成功：

```
╔═══════════════════════════════════════════╗
║  肺炎X光辅助诊断系统 — API 服务           ║
╚═══════════════════════════════════════════╝
加载模型: .../model/checkpoints/best_model.pth | 设备: cpu
模型加载完成！
 * Running on http://127.0.0.1:5002
```

### API 端点

| 方法 | 路径               | 说明             |
|------|--------------------|------------------|
| GET  | `/api/health`      | 健康检查         |
| POST | `/api/predict`     | 上传 X 光图片诊断 |
| GET  | `/api/class_info`  | 分类标签信息     |

---

## 前端启动

```bash
# 进入前端目录
cd /home/aq/深度学习课设/pneumonia-diagnosis/frontend

# （首次）安装依赖
npm install

# 启动开发服务器
npm run dev
```

看到以下输出即启动成功：

```
VITE v8.0.16  ready in 213 ms
➜  Local:   http://localhost:5173/
```

---

## 访问系统

打开浏览器访问 **http://localhost:5173**

前端 Vite 服务器已配置代理，所有 `/api/*` 请求会自动转发到后端 `http://localhost:5002`。

---

## 常见问题

### 端口被占用

```bash
# 查看占用端口的进程
fuser 5002/tcp
fuser 5173/tcp

# 强制释放端口
fuser -k 5002/tcp
fuser -k 5173/tcp
```

### 虚拟环境丢失或损坏

```bash
# 删除旧的 venv
rm -rf backend/venv

# 重新创建（如果提示 ensurepip 不可用，加 --without-pip）
python3 -m venv backend/venv

# 激活并安装 pip
source backend/venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python

# 安装依赖
pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### 模型权重缺失

模型文件应位于 `model/checkpoints/best_model.pth`（约 295MB）。  
如果缺失，需要将训练好的权重文件放到该路径。

### 前端代理报错 502

确保后端 Flask 服务正在运行（`curl http://localhost:5002/api/health`），  
Vite 的代理配置在 `frontend/vite.config.js` 中。

---

## 项目结构

```
pneumonia-diagnosis/
├── backend/
│   ├── app.py           # Flask API 入口
│   ├── predict.py       # 推理模块（模型加载 + 预测）
│   ├── venv/            # Python 虚拟环境
│   └── uploads/         # 上传文件临时目录
├── frontend/
│   ├── src/             # React 源码
│   ├── vite.config.js   # Vite 配置（含 API 代理）
│   └── package.json
├── model/
│   ├── model.py         # ResNet50 + CBAM 模型定义
│   └── checkpoints/
│       └── best_model.pth  # 训练好的权重
├── data/                # 数据集
└── STARTUP.md           # 本文件
```
