"""
推理模块：加载模型进行预测
"""
import os, sys, numpy as np
from PIL import Image
import torch, torch.nn.functional as F, torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from model import create_model

CLASS_NAMES = ["新冠肺炎", "正常", "肺部阴影", "病毒性肺炎"]
CLASS_COLORS = {"新冠肺炎": "#dc3545", "正常": "#28a745", "肺部阴影": "#fd7e14", "病毒性肺炎": "#ffc107"}
CLASS_ADVICE = {
    "新冠肺炎": "⚠️ 高疑似新冠肺炎，请立即进行核酸检测并隔离观察。",
    "正常": "✅ 肺部X光未见明显异常。",
    "肺部阴影": "⚠️ 检测到肺部磨玻璃影，建议进一步临床检查。",
    "病毒性肺炎": "⚠️ 疑似病毒性肺炎，建议进一步临床诊断。",
}


class Predictor:
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu")
        else:
            self.device = device

        if model_path is None:
            model_path = self._find_model()
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        self.model_path = model_path
        print(f"加载模型: {model_path} | 设备: {self.device}")
        self.model = self._load_model(model_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("模型加载完成！")

    def _find_model(self):
        search = [
            os.path.join(os.path.dirname(__file__), "..", "model", "checkpoints", "best_model.pth"),
            os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth"),
            "best_model.pth",
        ]
        for p in search:
            if os.path.exists(p):
                return os.path.abspath(p)
        return None

    def _load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model = create_model(len(CLASS_NAMES), pretrained=False, device=self.device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        return model

    def _preprocess(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        tensor = self._preprocess(image)
        outputs = self.model(tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probabilities))
        probs_dict = {CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))}
        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(float(probabilities[pred_idx]), 4),
            "probabilities": probs_dict,
            "class_color": CLASS_COLORS.get(CLASS_NAMES[pred_idx], "#6c757d"),
            "advice": CLASS_ADVICE.get(CLASS_NAMES[pred_idx], ""),
        }


_predictor = None


def get_predictor(model_path=None):
    global _predictor
    if _predictor is None:
        _predictor = Predictor(model_path)
    return _predictor


if __name__ == "__main__":
    predictor = get_predictor()
    if len(sys.argv) > 2:
        result = predictor.predict(sys.argv[2])
        print(f"预测: {result['prediction']} ({result['confidence']:.2%})")
