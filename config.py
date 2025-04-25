import os

class Config:
    # 基础配置
    DEBUG = True
    PORT = 8000
    HOST = '0.0.0.0'  # 允许从任何IP访问
    
    # 动态获取当前脚本所在目录的绝对路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据目录配置
    DATA_DIR = os.path.join(BASE_DIR, 'classify')
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BASE_DIR, 'app.log') 