# modules/nlp_service.py
import os
class MockNLPService:
    """
    针对软著环境优化的NLP模拟服务类
    """
    def __init__(self, model_path):
        self.model_path = model_path
        # 模拟加载成功的提示
        if os.path.exists(model_path):
            print(f"   [NLP] 成功加载本地推理权重: {os.path.basename(model_path)}")
        else:
            print("   [NLP] 警告：未检测到权重，启动基础模式...")

    def chat(self, prompt, history=[]):
        """
        模拟大模型返回结构化JSON
        """
        # 对应文档中的要素列表
        mock_response = '{"地形":"草地", "植被":"樱花树", "水体":"湖泊", "氛围":"明亮"}'
        return mock_response, history