import socket, struct, pickle, cv2, numpy as np, time, os, json

# ============================================================
# --- Configuration ---
# ============================================================
HOST, PORT = "172.20.240.1", 8485   # 👈 your Windows IP
SHOW_PREVIEW = True                  # ✅ Set True to show webcam in WSL

# --- Load class labels from JSON files ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

asl_labels_path = os.path.join(CONFIG_DIR, "labels_asl.json")
wlasl_labels_path = os.path.join(CONFIG_DIR, "labels_wlasl.json")

asl_labels = {}
wlasl_labels = {}

if os.path.exists(asl_labels_path):
    with open(asl_labels_path, "r") as f:
        asl_labels = json.load(f)
    print(f"📘 Loaded {len(asl_labels)} ASL labels")
else:
    print("⚠️ Missing labels_asl.json — using numeric indices")

if os.path.exists(wlasl_labels_path):
    with open(wlasl_labels_path, "r") as f:
        wlasl_labels = json.load(f)
    print(f"📗 Loaded {len(wlasl_labels)} WLASL labels")
else:
    print("⚠️ Missing labels_wlasl.json — using numeric indices")

# ============================================================
# --- Connect to Windows Host ---
# ============================================================
print(f"🔍 Connecting to Windows host {HOST}:{PORT}...")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

for attempt in range(10):
    try:
        client.connect((HOST, PORT))
        print("✅ Connected to host GPU inference server.")
        break
    except Exception as e:
        print(f"Retry {attempt+1}/10 failed: {e}")
        time.sleep(1)
else:
    raise RuntimeError("❌ Unable to connect to Windows host.")

print("📡 Listening for inference results...")

last_time = time.time()
frame_count = 0
fps = 0.0

# ============================================================
# --- Main loop ---
# ============================================================
try:
    while True:
        header = client.recv(4)
        if not header:
            print("❌ Connection closed.")
            break

        msg_len = struct.unpack("!I", header)[0]
        payload = b""
        while len(payload) < msg_len:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload += chunk

        result = pickle.loads(payload)
        label_idx = result.get("label_idx", -1)
        conf = result.get("confidence", 0.0)
        frame_bytes = result.get("frame", None)

        # --- Decode frame ---
        if frame_bytes is not None:
            jpg = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

            # Compute FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                frame_count = 0
                last_time = time.time()

            # --- Label lookup ---
            asl_label = asl_labels.get(str(label_idx), f"ASL_{label_idx}")
            wlasl_label = wlasl_labels.get(str(label_idx), f"WLASL_{label_idx}")

            # --- Console log ---
            print(f"🧠 ResNet: {asl_label:20s} | MViT: {wlasl_label:20s} | conf: {conf:.3f}")

            # --- Overlay ---
            if SHOW_PREVIEW:
                overlay_text = f"{asl_label} / {wlasl_label} ({conf:.2f}) | FPS: {fps:.1f}"
                cv2.putText(frame, overlay_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("WSL Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print(f"🧠 Prediction: {label_idx} (conf: {conf:.2f})")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")
finally:
    client.close()
    cv2.destroyAllWindows()
    print("✅ Connection closed.")
