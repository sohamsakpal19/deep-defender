import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

class HFImageModel:
    def __init__(self, model_id="prithivMLmods/Deepfake-Detect-Siglip2", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
       
        except (OSError, ValueError, EnvironmentError) as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face image processor for '{model_id}'. "
                "Please verify the model ID is correct, required files are available, "
                "and network connectivity is working."
            ) from exc

        try:
            self.model = SiglipForImageClassification.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face image model for '{model_id}'. "
                "Please verify the model ID is correct, required files are available, "
                "and network connectivity is working."
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        image: PIL.Image or numpy array (H, W, 3) in RGB
        Returns: logits, probs (softmax)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        return logits.squeeze(0), probs.squeeze(0)

    def id2label(self):
        return self.model.config.id2label