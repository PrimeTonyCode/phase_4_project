# Quantize BERT model for deployment
from transformers import BertForSequenceClassification
from optimum.pytorch import quantize_dynamic
import torch

MODEL_PATH = "model.safetensors"
QUANTIZED_MODEL_PATH = "model_quantized.safetensors"

# Load the original model
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Quantize the model (dynamic quantization)
quantized_model = quantize_dynamic(model, dtype=torch.qint8)

# Save the quantized model
quantized_model.save_pretrained(QUANTIZED_MODEL_PATH)
print(f"Quantized model saved to {QUANTIZED_MODEL_PATH}")
