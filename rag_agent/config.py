"""配置管理模块"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """应用配置类

    从环境变量中加载配置，提供类型安全的配置访问。
    """

    # API 配置
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    # 数据集配置
    DATASET_NAME: str = os.getenv("DATASET_NAME", "squad")
    DATASET_SPLIT: str = os.getenv("DATASET_SPLIT", "train")
    DATASET_SAMPLE_SIZE: int = int(os.getenv("DATASET_SAMPLE_SIZE", "100"))

    # 多数据集支持（优先级高于单一数据集）
    # 格式: "electrical_engineering,baa_industrial" 或 "all" 加载所有
    MULTI_DATASETS: str | None = os.getenv("MULTI_DATASETS", None)

    # 向量数据库配置
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 向量数据库持久化路径
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", str(Path(__file__).parent.parent / ".vectorstore"))

    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    DIAGNOSIS_TEMPLATE_ID: str = os.getenv("DIAGNOSIS_TEMPLATE_ID", "device_diagnosis")
    LATEX_COMPILATION_TIMEOUT: int = int(os.getenv("LATEX_COMPILATION_TIMEOUT", "120"))

    @classmethod
    def validate(cls) -> bool:
        """验证必要的配置是否存在

        Returns:
            bool: 配置验证通过返回 True

        Raises:
            ValueError: 当必要的配置缺失时
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        return True


config = Config()
