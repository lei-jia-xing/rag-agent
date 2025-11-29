"""配置管理模块"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """应用配置类"""

    # API 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    # 数据集配置
    DATASET_NAME = os.getenv("DATASET_NAME", "squad")
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
    DATASET_SAMPLE_SIZE = int(os.getenv("DATASET_SAMPLE_SIZE", "100"))

    # 向量数据库配置
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    @classmethod
    def validate(cls):
        """验证必要的配置是否存在"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        return True


config = Config()
