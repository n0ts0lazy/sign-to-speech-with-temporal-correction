import onnxruntime as ort
import numpy as np
import os

# --- Detect and select the best provider automatically ---
available_providers = ort.get_available_providers()
preferred_providers = []

if "TensorrtExecutionProvider" in available_providers:
    preferred_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
elif "CUDAExecutionProvider" in available_providers:
    preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    preferred_providers = ["CPUExecutionProvider"]

print(f"🔍 Using execution providers: {preferred_providers}")

# --- Model paths ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "model_inter")

resnet_path = os.path.join(MODEL_DIR, "resnet_asl_fp32.onnx")
mvit_path   = os.path.join(MODEL_DIR, "mvit_wlasl_fp32.onnx")

print("🧠 Loading models:")
print(f"  ResNet: {resnet_path}")
print(f"  MViT:   {mvit_path}")

# --- Load ONNX sessions ---
resnet_sess = ort.InferenceSession(resnet_path, providers=preferred_providers)
mvit_sess   = ort.InferenceSession(mvit_path, providers=preferred_providers)

# --- Core ensemble inference ---
def run_dual_inference(image, clip, alpha=0.5, softmax=True):
    """
    Runs inference on both ResNet (image) and MViT (clip) models,
    combines their outputs, and returns final probability scores.
    """
    # Run both models
    r_out = resnet_sess.run(None, {resnet_sess.get_inputs()[0].name: image})[0]
    m_out = mvit_sess.run(None, {mvit_sess.get_inputs()[0].name: clip})[0]

    if softmax:
        r_out = np.exp(r_out) / np.sum(np.exp(r_out), axis=1, keepdims=True)
        m_out = np.exp(m_out) / np.sum(np.exp(m_out), axis=1, keepdims=True)

    # Combine with weighted average
    probs = alpha * r_out + (1 - alpha) * m_out
    return probs

# --- Test run (optional) ---
if __name__ == "__main__":
    print("🚀 Running ensemble dry-run with dummy data...")
    dummy_image = np.random.rand(1, 3, 224, 224).astype(np.float32)
    dummy_clip  = np.random.rand(1, 3, 16, 224, 224).astype(np.float32)
    output = run_dual_inference(dummy_image, dummy_clip)
    print("✅ Output shape:", output.shape)
