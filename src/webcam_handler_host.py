import os, sys

# --- Hardcode CUDA and cuDNN paths for Windows ---
if sys.platform.startswith("win"):
    # Adjust these if your versions change
    cuda_ver = "v12.4"  # your active CUDA version
    cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.14\bin\12.9"
    cuda_path = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cuda_ver}\bin"

    for p in [cuda_path, cudnn_path]:
        if os.path.exists(p):
            os.add_dll_directory(p)
            print(f"🔗 Added DLL path: {p}")
        else:
            print(f"⚠️ Missing expected path: {p}")

import socket, struct, pickle, cv2, numpy as np, onnxruntime as ort
from collections import deque

HOST, PORT = "0.0.0.0", 8485
SEQ_LEN, IMG_SIZE = 16, 224
SHOW_PREVIEW = False  # ✅ Set True if you want to see the webcam on Windows

print("📸 Initializing webcam on Windows host...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam.")

# --- Select execution provider ---
providers = ort.get_available_providers()
if "CUDAExecutionProvider" in providers:
    EP = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("✅ Using GPU (CUDAExecutionProvider)")
else:
    EP = ["CPUExecutionProvider"]
    print("⚠️ GPU not available — using CPU")

# --- Load models ---
print("🧠 Loading ONNX models...")
resnet_sess = ort.InferenceSession("./models/model_inter/resnet_asl_fp32.onnx", providers=EP)
mvit_sess   = ort.InferenceSession("./models/model_inter/mvit_wlasl_fp32.onnx", providers=EP)
print("✅ Models loaded successfully")

frame_buffer = deque(maxlen=SEQ_LEN)

# --- Set up socket server ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"📡 Waiting for WSL to connect on port {PORT}...")

client, addr = server.accept()
print(f"🔗 Connected to {addr}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_np = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_buffer.append(img_np)

        # --- Show preview on Windows if desired ---
        if SHOW_PREVIEW:
            cv2.imshow("Webcam (Host)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(frame_buffer) < SEQ_LEN:
            continue

        resnet_in = np.expand_dims(img_np, axis=0)                    # (1,3,224,224)
        mvit_in   = np.expand_dims(np.stack(frame_buffer, axis=1), 0) # (1,3,T,224,224)

        resnet_out = resnet_sess.run(None, {resnet_sess.get_inputs()[0].name: resnet_in})[0]
        mvit_out   = mvit_sess.run(None, {mvit_sess.get_inputs()[0].name: mvit_in})[0]

        min_dim = min(resnet_out.shape[1], mvit_out.shape[1])
        probs = 0.5 * resnet_out[:, :min_dim] + 0.5 * mvit_out[:, :min_dim]
        pred_idx, confidence = int(np.argmax(probs)), float(np.max(probs))

        # --- Encode and send frame + results ---
        _, jpeg_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = jpeg_buf.tobytes()

        payload = pickle.dumps({
            "label_idx": pred_idx,
            "confidence": confidence,
            "frame": frame_bytes
        })
        client.sendall(struct.pack("!I", len(payload)) + payload)

except KeyboardInterrupt:
    print("\n🛑 Host stream stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    client.close()
    server.close()
    print("✅ Closed connections cleanly.")
