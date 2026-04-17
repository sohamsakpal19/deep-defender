import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

class HFAudioModel:
    def __init__(self, model_id="MelodyMachine/Deepfake-audio-detection-V2", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio classification processor for model '{model_id}'. "
                "Check that the model ID is correct, required files are available, "
                "and network connectivity/authentication to Hugging Face is working."
            ) from exc

        try:
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio classification model '{model_id}'. "
                "Check that the model ID is correct, required files are available, "
                "and network connectivity/authentication to Hugging Face is working."
            ) from exc
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, waveform, sample_rate):
        """
        waveform: 1D numpy array or torch tensor (mono)
        sample_rate: int
        Returns: logits, probs (softmax)
        """
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
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