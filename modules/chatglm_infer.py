# modules/nlp_service.py

class MockChatGLM:
    """
    模拟大模型推理类
    """

    def __init__(self, model_path):
        self.model_path = model_path
        print(f"[NLP] 成功从本地加载智能辨证模型库: {model_path}")

    def chat(self, tokenizer, prompt, history=[]):
        # 模拟大模型返回的JSON格式字符串
        # 对应文档中患者说的语音输入
        mock_response = '{"地形":"草地", "植被":"樱花树", "水体":"湖泊", "氛围":"明亮"}'
        return mock_response, history


def load_nlp_model(path):
    # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()

    tokenizer = "Local_Tokenizer_Stub"
    model = MockChatGLM(path)
    return tokenizer, model