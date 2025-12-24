# 🚀 Media Recommender System - 运行指南

## 目录
1. [快速开始](#快速开始)
2. [环境设置](#环境设置)
3. [本地开发运行](#本地开发运行)
4. [Docker部署](#docker部署)
5. [训练模型](#训练模型)
6. [运行测试](#运行测试)
7. [API使用示例](#api使用示例)
8. [常见问题](#常见问题)

---

## 快速开始

### 最简单的方式 (3步)

```bash
# 1. 解压项目
unzip media-recommender.zip
cd media-recommender

# 2. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 运行API服务
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问: http://localhost:8000/docs 查看API文档

---

## 环境设置

### 系统要求
- Python 3.9+
- CUDA 11.8+ (GPU训练，可选)
- Docker & Docker Compose (容器部署，可选)
- 8GB+ RAM (推荐16GB)

### 方法1: 使用pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 安装项目（开发模式）
pip install -e .

# 下载spaCy模型
python -m spacy download en_core_web_sm
```

### 方法2: 使用conda

```bash
# 创建conda环境
conda create -n media-rec python=3.10
conda activate media-rec

# 安装PyTorch (选择适合你CUDA版本的)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# 或 CPU版本
conda install pytorch torchvision cpuonly -c pytorch

# 安装其他依赖
pip install -r requirements.txt
pip install -e .
```

### 方法3: 使用Makefile

```bash
# 安装开发依赖
make dev-install

# 或只安装运行时依赖
make install
```

---

## 本地开发运行

### 1. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置 (可选，使用默认值也可以运行)
nano .env
```

`.env` 主要配置项:
```env
# API配置
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# 模型配置
MODEL_CHECKPOINT_DIR=./checkpoints
BERT_MODEL_NAME=bert-base-uncased

# 数据库 (本地开发可跳过)
DATABASE_URL=postgresql://user:pass@localhost:5432/recommender
REDIS_URL=redis://localhost:6379/0
```

### 2. 启动API服务

```bash
# 开发模式 (自动重载)
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 或使用Makefile
make serve

# 生产模式 (多worker)
make serve-prod
```

### 3. 验证服务运行

```bash
# 健康检查
curl http://localhost:8000/health

# 应该返回:
# {"status": "healthy", "model_loaded": true, ...}
```

### 4. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Docker部署

### 方法1: 只运行API服务

```bash
# 构建镜像
docker build -t media-recommender:latest .

# 运行容器
docker run -d \
  --name recommender-api \
  -p 8000:8000 \
  -e DEBUG=false \
  media-recommender:latest
```

### 方法2: 完整服务栈 (推荐)

```bash
# 启动所有服务 (API + PostgreSQL + Redis + Elasticsearch + 监控)
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 查看服务状态
docker-compose ps
```

服务端口:
| 服务 | 端口 | 说明 |
|------|------|------|
| API | 8000 | 推荐服务 |
| PostgreSQL | 5432 | 数据库 |
| Redis | 6379 | 缓存 |
| Elasticsearch | 9200 | 搜索 |
| Prometheus | 9090 | 监控指标 |
| Grafana | 3000 | 监控面板 |
| MLflow | 5001 | 实验追踪 |
| Jaeger | 16686 | 分布式追踪 |

### 方法3: 开发环境Docker

```bash
# 使用开发配置
docker-compose -f docker-compose.yml up -d postgres redis

# API在本地运行，数据库用Docker
python -m uvicorn src.api.main:app --reload
```

---

## 训练模型

### 1. 使用综合训练脚本（推荐）

```bash
# 训练所有模型
python scripts/train_all.py

# 训练特定模型
python scripts/train_all.py --models ncf,bert4rec --epochs 20

# 使用GPU训练
python scripts/train_all.py --device cuda --data-size large

# 训练后评估
python scripts/train_all.py --epochs 10 --eval
```

### 2. 训练单个模型

```bash
# 协同过滤
python -m src.training.train_collaborative --epochs 10

# 序列模型
python -m src.training.train_sequential --model lstm --epochs 10
python -m src.training.train_sequential --model sasrec --epochs 10
```

### 3. 快速验证Demo

```bash
# 设置Python路径
export PYTHONPATH=.

# 运行demo脚本
python scripts/demo.py
```

### 4. 完整训练示例

```python
# train_example.py
import torch
from torch.utils.data import DataLoader
from src.models import CollaborativeConfig, NeuralCollaborativeFiltering
from src.data import InteractionDataset, DataProcessor
from src.training import Trainer, TrainingConfig

# 1. 加载数据
processor = DataProcessor()
train_data, val_data, test_data = processor.load_and_split('data/interactions.csv')

# 2. 创建Dataset
train_dataset = InteractionDataset(train_data, processor.user_map, processor.item_map)
val_dataset = InteractionDataset(val_data, processor.user_map, processor.item_map)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# 3. 初始化模型
config = CollaborativeConfig(
    num_users=len(processor.user_map),
    num_items=len(processor.item_map),
    embedding_dim=64
)
model = NeuralCollaborativeFiltering(config)

# 4. 训练
train_config = TrainingConfig(
    epochs=50,
    learning_rate=0.001,
    early_stopping_patience=5
)

trainer = Trainer(model, train_config, device='cuda')
trainer.train(train_loader, val_loader)

# 5. 保存模型
torch.save(model.state_dict(), 'checkpoints/ncf_model.pt')
```

运行:
```bash
python train_example.py
```

---

## 运行测试

### 运行所有测试

```bash
# 使用pytest
pytest tests/ -v

# 或使用Makefile
make test
```

### 运行特定测试

```bash
# 只测试模型
pytest tests/test_models.py -v

# 只测试API
pytest tests/test_api.py -v

# 只测试数据处理
pytest tests/test_data.py -v
```

### 测试覆盖率

```bash
# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 查看报告
open htmlcov/index.html
```

### 快速验证模型

```python
# quick_test.py
import torch
from src.models import (
    CollaborativeConfig, NeuralCollaborativeFiltering,
    TransformerRecConfig, BERT4Rec,
    GNNConfig, LightGCN
)

print("Testing models...")

# Test NCF
config = CollaborativeConfig(num_users=100, num_items=500)
ncf = NeuralCollaborativeFiltering(config)
users = torch.randint(0, 100, (32,))
items = torch.randint(0, 500, (32,))
scores = ncf(users, items)
print(f"✓ NCF output: {scores.shape}")

# Test BERT4Rec
config = TransformerRecConfig(num_items=500, max_seq_len=50)
bert4rec = BERT4Rec(config)
sequences = torch.randint(0, 500, (4, 50))
logits = bert4rec(sequences)
print(f"✓ BERT4Rec output: {logits.shape}")

# Test LightGCN
config = GNNConfig(num_users=100, num_items=500)
lightgcn = LightGCN(config)
print(f"✓ LightGCN initialized")

print("\n✅ All models working correctly!")
```

运行:
```bash
python quick_test.py
```

---

## API使用示例

### 1. 获取推荐

```bash
# 获取用户推荐
curl -X POST "http://localhost:8000/api/v1/recommendations/123" \
  -H "Content-Type: application/json" \
  -d '{
    "n_recommendations": 10,
    "exclude_items": [1, 2, 3]
  }'
```

响应:
```json
{
  "user_id": "123",
  "recommendations": [
    {"item_id": 456, "score": 0.95, "source": "hybrid"},
    {"item_id": 789, "score": 0.92, "source": "collaborative"}
  ],
  "generated_at": "2024-01-15T10:30:00Z"
}
```

### 2. 记录用户交互

```bash
curl -X POST "http://localhost:8000/api/v1/interactions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "item_id": 456,
    "interaction_type": "click",
    "timestamp": "2024-01-15T10:30:00Z"
  }'
```

### 3. 获取相似物品

```bash
curl "http://localhost:8000/api/v1/items/456/similar?n=5"
```

### 4. Python客户端示例

```python
import requests

class RecommenderClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_recommendations(self, user_id, n=10):
        response = requests.post(
            f"{self.base_url}/api/v1/recommendations/{user_id}",
            json={"n_recommendations": n}
        )
        return response.json()
    
    def log_interaction(self, user_id, item_id, interaction_type="click"):
        response = requests.post(
            f"{self.base_url}/api/v1/interactions",
            json={
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type
            }
        )
        return response.json()

# 使用
client = RecommenderClient()
recs = client.get_recommendations(user_id="123", n=10)
print(recs)
```

---

## 常见问题

### Q1: 缺少依赖包

```bash
# 安装缺失的包
pip install <package_name>

# 或重新安装所有依赖
pip install -r requirements.txt --force-reinstall
```

### Q2: CUDA/GPU问题

```bash
# 检查PyTorch是否检测到GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，安装CUDA版PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 内存不足

```python
# 在训练时减少batch_size
train_config = TrainingConfig(batch_size=64)  # 默认256

# 或使用梯度累积
train_config = TrainingConfig(
    batch_size=64,
    gradient_accumulation_steps=4
)
```

### Q4: 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8000

# 使用其他端口
python -m uvicorn src.api.main:app --port 8001
```

### Q5: Docker构建失败

```bash
# 清理Docker缓存
docker system prune -a

# 重新构建
docker-compose build --no-cache
```

### Q6: 模型加载失败

```python
# 检查checkpoint路径
import os
print(os.path.exists('checkpoints/model.pt'))

# 使用CPU加载GPU训练的模型
model.load_state_dict(
    torch.load('model.pt', map_location='cpu')
)
```

---

## 项目结构速览

```
media-recommender/
├── src/
│   ├── models/          # 50+种ML模型
│   ├── data/            # 数据处理
│   ├── training/        # 训练逻辑
│   ├── api/             # FastAPI服务
│   └── utils/           # 工具函数
├── tests/               # 测试文件
├── configs/             # 配置文件
├── docker/              # Docker配置
├── scripts/             # 部署脚本
├── requirements.txt     # 依赖
├── Dockerfile          
├── docker-compose.yml   
└── Makefile            # 常用命令
```

---

## 下一步

1. **本地测试**: `make test`
2. **启动服务**: `make serve`
3. **Docker部署**: `docker-compose up -d`
4. **训练自己的模型**: 准备数据，运行训练脚本
5. **查看监控**: http://localhost:3000 (Grafana)

如有问题，请检查日志:
```bash
# API日志
docker-compose logs -f api

# 或本地运行时查看终端输出
```
