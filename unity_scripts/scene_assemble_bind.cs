using System;
using System.Collections.Generic;
using System.Linq;

// 语音生成个性化场景+康复动作强制绑定（技术路线模块3）
namespace VRSceneActionBind
{
    // 场景要素类（语音解析结果）
    public class SceneElements
    {
        public string 地形 { get; set; }
        public string 植被 { get; set; }
        public string 水体 { get; set; }
        public string 氛围 { get; set; }
    }

    // 3D组件库（技术路线3.1，单个组件≤5MB）
    public class SceneComponentLib
    {
        // 地形组件
        public Dictionary<string, string> 地形组件 = new Dictionary<string, string>()
        {
            {"草地", "Prefabs/地形/草地.prefab"},
            {"沙地", "Prefabs/地形/沙地.prefab"},
            {"泥土", "Prefabs/地形/泥土.prefab"}
        };

        // 植被组件
        public Dictionary<string, string> 植被组件 = new Dictionary<string, string>()
        {
            {"樱花树", "Prefabs/植被/樱花树.prefab"},
            {"柳树", "Prefabs/植被/柳树.prefab"},
            {"小草", "Prefabs/植被/小草.prefab"}
        };

        // 水体组件
        public Dictionary<string, string> 水体组件 = new Dictionary<string, string>()
        {
            {"湖泊", "Prefabs/水体/湖泊.prefab"},
            {"小溪", "Prefabs/水体/小溪.prefab"},
            {"池塘", "Prefabs/水体/池塘.prefab"}
        };

        // 氛围组件
        public Dictionary<string, string> 氛围组件 = new Dictionary<string, string>()
        {
            {"明亮", "Prefabs/氛围/明亮光照.prefab"},
            {"柔和", "Prefabs/氛围/柔和光照.prefab"}
        };

        // 康复动作-场景道具映射（技术路线3.2）
        public Dictionary<string, Dictionary<string, string>> 动作道具映射 = new Dictionary<string, Dictionary<string, string>>()
        {
            {
                "上肢伸展", new Dictionary<string, string>()
                {
                    {"樱花树", "樱花花瓣"},
                    {"柳树", "柳叶"},
                    {"小草", "露珠"}
                }
            },
            {
                "抓握", new Dictionary<string, string>()
                {
                    {"湖泊", "湖边野果"},
                    {"小溪", "溪边长草"},
                    {"池塘", "池塘莲子"}
                }
            }
        };
    }

    // 核心绑定引擎（含约束校验，技术路线3.2/3.3）
    public class SceneActionBindEngine
    {
        private SceneComponentLib _componentLib = new SceneComponentLib();

        // 1. 个性化场景拼装（技术路线3.1）
        public string AssembleScene(SceneElements elements)
        {
            // 按“地形→水体→植被→氛围”层级拼装，加载延迟≤2s
            string sceneLog = "";
            sceneLog += $"1. 加载地形组件：{_componentLib.地形组件[elements.地形]}\n";
            sceneLog += $"2. 加载水体组件：{_componentLib.水体组件[elements.水体]}\n";
            sceneLog += $"3. 加载植被组件：{_componentLib.植被组件[elements.植被]}\n";
            sceneLog += $"4. 加载氛围组件：{_componentLib.氛围组件[elements.氛围]}\n";

            Console.WriteLine("模块3：个性化场景拼装完成\n" + sceneLog);
            return sceneLog;
        }

        // 2. 绑定约束引擎（技术路线3.3，校验+自动调整）
        public bool ValidatePropParams(string actionName, ref float propHeight, ref float propSpeed, int muscleGrade)
        {
            // 康复动作要素约束阈值（技术路线3.2）
            Dictionary<string, Dictionary<string, float>> actionConstraints = new Dictionary<string, Dictionary<string, float>>()
            {
                {
                    "上肢伸展", new Dictionary<string, float>()
                    {
                        {"maxHeight", 1.2f},  // 最大高度1.2m
                        {"maxSpeed", 0.2f},   // 最大移动速度0.2m/s
                        {"minAngle", 0f},     // 最小角度0°
                        {"maxAngle", 90f}     // 最大角度90°
                    }
                },
                {
                    "抓握", new Dictionary<string, float>()
                    {
                        {"maxHeight", 1.0f},  // 最大高度1.0m
                        {"maxSpeed", 0.15f},  // 最大移动速度0.15m/s
                        {"minGripForce", 0f}, // 最小握力0N
                        {"maxGripForce", 50f} // 最大握力50N
                    }
                }
            };

            var constraints = actionConstraints[actionName];
            bool isValid = true;

            // 适配肌力分级的高度约束
            float muscleAdaptedMaxHeight = muscleGrade switch
            {
                1 => constraints["maxHeight"] * 0.4f,  // 肌力1级→40%最大高度
                2 => constraints["maxHeight"] * 0.6f,  // 肌力2级→60%最大高度
                3 => constraints["maxHeight"] * 0.8f,  // 肌力3级→80%最大高度
                _ => constraints["maxHeight"]          // 肌力4-5级→100%最大高度
            };

            // 校验并修正道具高度
            if (propHeight > muscleAdaptedMaxHeight)
            {
                Console.WriteLine($"约束校验：道具高度超标（{propHeight}m＞{muscleAdaptedMaxHeight}m），自动修正");
                propHeight = muscleAdaptedMaxHeight;
                isValid = false;
            }

            // 校验并修正道具速度
            if (propSpeed > constraints["maxSpeed"])
            {
                Console.WriteLine($"约束校验：道具速度超标（{propSpeed}m/s＞{constraints["maxSpeed"]}m/s），自动修正");
                propSpeed = constraints["maxSpeed"] * 0.8f;
                isValid = false;
            }

            return isValid;
        }

        // 3. 康复动作强制绑定（技术路线3.2）
        public void BindActionToScene(string actionName, SceneElements elements, int muscleGrade)
        {
            // 获取适配场景的康复道具
            string propType = _componentLib.动作道具映射[actionName][elements.植被];

            // 初始道具参数
            float propHeight = muscleGrade switch
            {
                1 => 0.5f,
                2 => 0.8f,
                _ => 1.2f
            };
            float propSpeed = 0.1f;

            // 执行约束校验与自动调整
            bool isParamsValid = ValidatePropParams(actionName, ref propHeight, ref propSpeed, muscleGrade);

            // 绑定动作核心要素
            string bindInfo = $"动作绑定结果：\n" +
                             $" - 康复动作：{actionName}\n" +
                             $" - 场景道具：{propType}\n" +
                             $" - 道具高度：≤{propHeight}m（适配肌力{muscleGrade}级）\n" +
                             $" - 移动轨迹：{actionName == "上肢伸展" ? "矢状面" : "抓握范围"}\n" +
                             $" - 动作幅度：{actionName == "上肢伸展" ? "0-90°" : "0-45°"}\n" +
                             $" - 移动速度：{propSpeed}m/s\n" +
                             $" - 动作频率：1次/3s";
            Console.WriteLine(bindInfo);
        }

        // 4. 动作完成度校验（技术路线2.2/3.2）
        public bool CheckActionCompletion(float actualAngle, float maxAngle)
        {
            float completion = actualAngle / maxAngle;
            Console.WriteLine($"动作完成度：{completion:P2}（≥80%为有效）");
            return completion >= 0.8f;
        }
    }

    // 语音解析场景要素（技术路线3.1，基于ChatGLM-6B-int4）
    public class VoiceSceneParser
    {
        // 模拟语音转文字（实际对接VR终端麦克风）
        public string VoiceToText(string voiceInput)
        {
            Console.WriteLine($"\n接收患者语音输入：{voiceInput}");
            return voiceInput;
        }

        // 解析场景要素（结构化JSON输出）
        public SceneElements ParseSceneElements(string voiceInput)
        {
            string text = VoiceToText(voiceInput);

            // 调用ChatGLM-6B-int4推理服务
            var requestBody = new { text = text };
            var jsonContent = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");
            var response = _httpClient.PostAsync(_glmApiUrl, jsonContent).Result;
            response.EnsureSuccessStatusCode();

            // 解析大模型返回的结构化结果
            var glmResponse = response.Content.ReadFromJsonAsync<SceneElements>().Result;

            Console.WriteLine($"场景要素解析结果：地形={glmResponse.地形}，植被={glmResponse.植被}，水体={glmResponse.水体}，氛围={glmResponse.氛围}");
            return glmResponse;
        }
    }

    // 测试主程序
    class Program
    {
        static void Main(string[] args)
        {
            // 模拟流程：语音输入→要素解析→场景拼装→动作绑定→完成度校验
            VoiceSceneParser parser = new VoiceSceneParser();
            SceneActionBindEngine bindEngine = new SceneActionBindEngine();

            // 1. 患者语音输入场景
            string patientVoice = "我想在有樱花树的湖边草地训练";
            SceneElements elements = parser.ParseSceneElements(patientVoice);

            // 2. 个性化场景拼装
            bindEngine.AssembleScene(elements);

            // 3. 康复动作绑定（动作识别结果：上肢伸展，肌力2级）
            bindEngine.BindActionToScene("上肢伸展", elements, 2);

            // 4. 动作完成度校验（实际角度72°，最大90°）
            bool isEffective = bindEngine.CheckActionCompletion(72f, 90f);
            Console.WriteLine($"动作有效性判定：{isEffective}（有效动作→继续训练/无效动作→语音引导）");
        }
    }
}