# 1. 选择基础镜像：直接用官方 PyTorch 镜像，省去安装 CUDA 和 Python 的麻烦
# 建议根据你的项目需求选择版本，例如 Qwen2.5 通常需要较新的 CUDA
# FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
FROM nvcr.io/nvidia/pytorch:24.07-py3

# 2. 设置工作目录（在容器内部的路径）
WORKDIR /workspace

# 3. 安装系统级依赖（如 git, vim, 或者量化需要的库）
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 预先安装 Python 依赖（利用 Docker 缓存机制，这样以后改代码不需要重新装包）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 拷贝你的项目代码到镜像中
# 注意：大型模型权重文件建议不要 COPY 进去，而是运行时挂载
COPY . /workspace

# 6. 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认启动命令
CMD ["bash"]