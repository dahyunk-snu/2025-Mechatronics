# 1) 필요 모듈 임포트
import streamlit as st
import threading
import socket
import json
import numpy as np
import torch
import torch.nn as nn
from gtts import gTTS
import tempfile
import time
import os # For deleting temp files
from streamlit_autorefresh import st_autorefresh

# --- Global Variables ---
latest_grid = None
latest_prediction = None
audio_bytes = None

model = None
device = torch.device("cpu") # Keep as CPU for broader compatibility
lock = threading.Lock()

# ------------------------------------------------
# 1. 브라유 인식용 PyTorch 네트워크 클래스 정의
#    (모델 로드할 때 동일 클래스 필요)
# ------------------------------------------------
class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input (N, 1, 8, 10) -> conv1 (N, 32, 8, 10) -> pool1 (N, 32, 4, 5)
        # -> conv2 (N, 64, 4, 5) -> pool2 (N, 64, 2, 2) [5//2=2]
        # Flattened features: 64 * 2 * 2 = 256
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6) # 6 possible Braille dots for a single character

    def forward(self, x):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1) # Flatten
        out = self.relu3(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ------------------------------------------------
# 3. TCP 소켓 리스너 (백그라운드 쓰레드)
#    - Arduino(ESP-01)가 보내는 JSON {"grid": [[...], ...]} 수신
# ------------------------------------------------
def socket_listener(host='0.0.0.0', port=5001):
    """
    1) 서버 소켓 열어서 클라이언트 연결(Arduino) 대기
    2) JSON payload를 모두 수신한 뒤, 파싱 → (8,10) 형태로 변환
    3) torch.Tensor 형태 (1,1,8,10)로 reshape & 정규화(255.0) 후 latest_grid에 저장
    """
    global latest_grid

    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv_sock.bind((host, port))
        srv_sock.listen(1)  # Allow only one Arduino(ESP-01) connection at a time
    except OSError as e:
        # st.error(f"Error binding socket: {e}. Port {port} might be in use or permissions issue.") # Can't use st.error in a non-Streamlit thread directly
        print(f"Error binding socket: {e}. Port {port} might be in use or permissions issue.")
        return # Exit the thread if binding fails

    while True:
        try:
            conn, addr = srv_sock.accept()
            data_bytes = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data_bytes += chunk
            conn.close()

            payload = json.loads(data_bytes.decode('utf-8'))
            grid_list = payload.get("grid", None)
            if grid_list is not None:
                arr = np.array(grid_list, dtype=np.float32)  # (8,10) shape
                if arr.shape == (8, 10):
                    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # PyTorch tensor: (1,1,8,10)
                    tensor = tensor / 255.0 # Normalize: scale 0-255 to 0-1
                    with lock:
                        latest_grid = tensor.clone()
        except Exception as e:
            # Ignore JSON parsing errors, shape errors, connection errors, etc.
            # print(f"Socket listener error: {e}") # For debugging
            pass

        time.sleep(0.05)  # Prevent excessive CPU usage

# ------------------------------------------------
# 4. 클래스 이름(인덱스 ↔ 한글 음절) 매핑 함수
#    - 모델 학습 시 사용된 순서와 **완전히 동일하게** 유지해야 함
# ------------------------------------------------
@st.cache_resource
def load_class_names():
    # Example: Assumed 40 Korean Braille syllables were trained
    # **NEVER CHANGE THE ORDER!**
    return [
        "가","나","다","라","마","바","사","아","자","차",
        "카","타","파","하","거","너","더","러","머","버",
        "서","어","저","처","커","터","퍼","허","고","노",
        "도","로","모","보","소","오","조","초","코","토"
    ]

# ------------------------------------------------
# 5. Streamlit 메인 함수
# ------------------------------------------------
def main():
    global model, latest_grid, latest_prediction, audio_bytes

    st.set_page_config(
        page_title="점자 AI 번역기",
        layout="wide",
        initial_sidebar_state="auto" # 'auto' for mobile, collapses by default
    )

    # --- Braille-Inspired, Modern & Sophisticated Custom CSS ---
    st.markdown(
        """
        <style>
            /* --- Global Styles --- */
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'AppleSDGothicNeo-Regular', 'Spoqa Han Sans Neo', 'Segoe UI', 'Malgun Gothic', sans-serif;
                background: #f0f2f6; /* Soft, light background */
                color: #2e3b4e; /* Dark text for high contrast */
            }

            /* --- Dot Pattern Background (for a Braille feel) --- */
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: radial-gradient(circle at 10% 20%, rgba(204,204,204,0.1) 1px, transparent 1px),
                                  radial-gradient(circle at 80% 90%, rgba(204,204,204,0.1) 1px, transparent 1px);
                background-size: 20px 20px; /* Adjust dot spacing */
                opacity: 0.7; /* Subtle opacity */
                z-index: -1;
            }

            /* --- Sidebar Styles --- */
            div[data-testid="stSidebar"] {
                background: #ffffff;
                padding-top: 25px;
                box-shadow: 4px 0 15px rgba(0,0,0,0.08);
                border-right: 1px solid #e5e5e5;
            }
            div[data-testid="stSidebar"] h2 {
                color: #1a2a3a;
                font-weight: 700;
                margin-bottom: 20px;
                padding-left: 20px;
                font-size: 1.5rem;
            }
            div[data-testid="stSidebar"] h3 {
                color: #4a6572;
                font-weight: 600;
                margin-top: 25px;
                margin-bottom: 15px;
                padding-left: 20px;
                font-size: 1.15rem;
            }
            div[data-testid="stSidebar"] .stButton > button {
                width: calc(100% - 40px);
                margin: 0 20px;
                border-radius: 10px; /* Slightly less rounded */
                border: none;
                color: white;
                background: #3498db; /* A clear, friendly blue for action */
                padding: 12px 0;
                font-size: 1rem;
                font-weight: 600;
                box-shadow: 0 4px 10px rgba(52,152,219,0.3);
                transition: all 0.3s ease-in-out;
            }
            div[data-testid="stSidebar"] .stButton > button:hover {
                background: #2980b9;
                box-shadow: 0 6px 12px rgba(52,152,219,0.4);
                transform: translateY(-2px);
            }
            div[data-testid="stSidebar"] .stAlert {
                margin: 20px;
                border-radius: 10px;
                font-size: 0.95rem;
            }
            div[data-testid="stSidebar"] .stSelectbox {
                padding: 0 20px;
                margin-top: 10px;
                margin-bottom: 20px;
            }
            div[data-testid="stSidebar"] .stSelectbox label {
                font-size: 1rem;
                color: #5a5a5a;
                margin-bottom: 8px;
            }

            /* --- Main Content Area --- */
            .main .block-container {
                padding-top: 3rem;
                padding-right: 2.5rem;
                padding-left: 2.5rem;
                padding-bottom: 3rem;
                max-width: 1200px;
                margin: auto;
            }

            /* Title - Bold and impactful */
            h1 {
                color: #1a2a3a;
                text-align: center;
                font-size: 3.2rem;
                margin-bottom: 0.75rem;
                font-weight: 800;
                letter-spacing: -1.5px;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.08);
            }
            /* Sub-headers for sections - Clear and concise */
            h3 {
                color: #4a6572;
                font-size: 1.8rem;
                margin-top: 2.5rem;
                margin-bottom: 1.5rem;
                font-weight: 700;
                position: relative;
                padding-bottom: 0.75rem;
                text-align: center;
            }
            h3::after { /* Underline effect for sub-headers */
                content: '';
                position: absolute;
                left: 50%;
                bottom: 0;
                transform: translateX(-50%);
                width: 70px; /* Slightly wider */
                height: 4px; /* Thicker */
                background-color: #3498db; /* Accent color */
                border-radius: 2px;
            }

            /* Introduction text */
            .intro-text {
                font-size: 1.1rem;
                line-height: 1.6;
                color: #5f6f7d;
                text-align: center;
                margin-bottom: 3rem;
                max-width: 750px;
                margin-left: auto;
                margin-right: auto;
            }
            .intro-text strong {
                color: #3498db; /* Accent color for emphasis */
            }

            /* Separator */
            hr {
                border: none;
                border-top: 1px solid #e0e0e0;
                margin: 3.5rem 0;
            }

            /* --- Content Cards/Containers --- */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                background-color: #ffffff;
                border-radius: 20px; /* More rounded */
                box-shadow: 0 10px 30px rgba(0,0,0,0.12); /* Deeper, softer shadow */
                padding: 2.5rem;
                margin-bottom: 2.5rem;
                transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            }
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
                transform: translateY(-7px); /* More pronounced lift */
                box-shadow: 0 15px 40px rgba(0,0,0,0.18);
            }

            /* Grid Image - Enhanced Braille Dot Visualization */
            div[data-testid="stImage"] {
                display: flex;
                flex-direction: column; /* To center caption below image */
                align-items: center;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }
            div[data-testid="stImage"] img {
                border: 5px solid #aebac8; /* A sophisticated, cool gray border */
                border-radius: 15px; /* More rounded image container */
                box-shadow: 0px 10px 25px rgba(0,0,0,0.15); /* Soft shadow */
                max-width: 100%;
                height: auto;
                object-fit: contain;
                /* --- Custom Dot Visualization for Braille Grid (Advanced) --- */
                /* This part is illustrative; direct pixel manipulation for "dots" isn't feasible with st.image
                   but a custom component or creative use of HTML/SVG might achieve this */
            }
            div[data-testid="stImage"] .stImageCaption {
                font-size: 0.95rem;
                color: #6a7c8c;
                text-align: center;
                margin-top: 1rem;
                font-style: italic;
            }

            /* Prediction Result - Bold and clear */
            .prediction-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 3rem 2.5rem;
                background: #e0eaff; /* Soft, inviting blue background */
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: 1px solid #c0d9ff;
                margin-top: 2rem;
                position: relative;
                overflow: hidden;
            }
            /* Subtle dot pattern inside prediction container */
            .prediction-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: radial-gradient(circle at 10% 20%, rgba(52,152,219,0.05) 1px, transparent 1px),
                                  radial-gradient(circle at 80% 90%, rgba(52,152,219,0.05) 1px, transparent 1px);
                background-size: 25px 25px;
                z-index: 0;
            }
            .prediction-container h2 {
                font-size: 5.5rem; /* Massive for impact */
                color: #2980b9; /* Deep blue for the core result */
                font-weight: 900; /* Extra bold */
                margin-bottom: 0.75rem;
                line-height: 1;
                text-shadow: 3px 3px 8px rgba(0,0,0,0.15);
                z-index: 1; /* Bring text above background pattern */
            }
            .prediction-container p {
                font-size: 1.5rem;
                color: #5f6f7d;
                margin-bottom: 1.5rem;
                font-weight: 500;
                z-index: 1;
            }

            /* Audio Player */
            audio {
                width: 95%;
                margin-top: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                background-color: #fcfcfc;
                z-index: 1;
            }

            /* Info/Warning Messages */
            div[data-testid="stAlert"] {
                border-radius: 12px;
                margin-top: 1.8rem;
                margin-bottom: 1.8rem;
                padding: 1.2rem 1.8rem;
                font-size: 1.05rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .st-emotion-cache-1f879j6 p {
                font-size: 1.05rem !important;
            }
            .st-emotion-cache-1f879j6.stAlert-info {
                background-color: #e6f7ff;
                border-left: 6px solid #1890ff;
                color: #1890ff;
            }
            .st-emotion-cache-1f879j6.stAlert-warning {
                background-color: #fffbe6;
                border-left: 6px solid #faad14;
                color: #faad14;
            }
            .st-emotion-cache-1f879j6.stAlert-success {
                background-color: #f6ffed;
                border-left: 6px solid #52c41a;
                color: #52c41a;
            }

            /* Spinner */
            .stSpinner > div > div {
                color: #3498db; /* Accent blue for spinner */
            }

            /* Streamlit defaults override for cleaner spacing */
            .st-emotion-cache-h5rgjs { /* block-container after first h1 */
                padding-top: 0;
            }
            .st-emotion-cache-1hm31g7 { /* stcolumns wrapper */
                gap: 3rem; /* More space between columns on desktop */
            }

            /* --- Mobile Specific Adjustments --- */
            @media (max-width: 768px) {
                h1 {
                    font-size: 2.5rem;
                }
                h3 {
                    font-size: 1.5rem;
                    margin-top: 1.8rem;
                    margin-bottom: 1rem;
                }
                .main .block-container {
                    padding-top: 1.5rem;
                    padding-right: 0.8rem;
                    padding-left: 0.8rem;
                    padding-bottom: 1.5rem;
                }
                .intro-text {
                    font-size: 1rem;
                    margin-bottom: 2rem;
                }
                hr {
                    margin: 2.5rem 0;
                }
                div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                    margin-bottom: 1.5rem;
                    padding: 1.5rem;
                    border-radius: 15px; /* Slightly less rounded for small screens */
                    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                }
                div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                }
                .prediction-container {
                    padding: 2.5rem 1.5rem;
                    border-radius: 15px;
                }
                .prediction-container h2 {
                    font-size: 4rem; /* Smaller for mobile */
                }
                .prediction-container p {
                    font-size: 1.2rem;
                }
                audio {
                    width: 100%;
                }
                div[data-testid="stSidebar"] {
                    padding-top: 15px;
                }
                div[data-testid="stSidebar"] .stButton > button {
                    width: calc(100% - 30px);
                    margin: 0 15px;
                    padding: 10px 0;
                    font-size: 0.95rem;
                    border-radius: 8px;
                }
                div[data-testid="stSidebar"] .stAlert {
                    margin: 15px;
                }
                body::before { /* Smaller dots for mobile background */
                    background-size: 15px 15px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # --- End Custom CSS ---

    st.title("⚫️ 오목")
    st.markdown(
        """
        <p class='intro-text'>
            손 끝에 정보를 담는 <b>실시간 점자 번역기</b>입니다.<br>
            실시간 점자 데이터를 <b>AI 모델</b>이 즉시 분류하고,<br>
            인식된 한글 문자를 <b>음성</b>으로 들려줍니다.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---") # Visual separator

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ 앱 설정")
    st.sidebar.markdown("여기서 모델 관리 및 음성 합성 설정을 변경할 수 있습니다.")

    # Model Load/Reload Section
    st.sidebar.markdown("### 🧠 모델 관리")
    if st.sidebar.button("모델 재로드"):
        with st.spinner("모델을 다시 로드 중..."): # Spinner appears in main area
            model = BrailleCNN()
            try:
                model.load_state_dict(torch.load("braille_cnn_model.pth", map_location=device))
                model.to(device)
                model.eval()
                st.sidebar.success("모델이 성공적으로 재로드되었습니다!")
            except FileNotFoundError:
                st.sidebar.error("⚠️ 'braille_cnn_model.pth' 파일을 찾을 수 없습니다. 모델 파일이 올바른 위치에 있는지 확인하세요.")
            except Exception as e:
                st.sidebar.error(f"⚠️ 모델 로드 중 오류 발생: {e}")

    if model is None:
        with st.spinner("초기 모델 로드 중..."): # Spinner appears in main area
            model = BrailleCNN()
            try:
                model.load_state_dict(torch.load("braille_cnn_model.pth", map_location=device))
                model.to(device)
                model.eval()
                st.sidebar.success("모델이 성공적으로 로드되었습니다!")
            except FileNotFoundError:
                st.sidebar.error("⚠️ 'braille_cnn_model.pth' 파일을 찾을 수 없습니다. 모델 파일이 올바른 위치에 있는지 확인하세요.")
            except Exception as e:
                st.sidebar.error(f"⚠️ 모델 로드 중 오류 발생: {e}")


    # TTS Language Selection Section
    st.sidebar.markdown("### 🔊 음성 합성 설정")
    tts_lang_option = st.sidebar.selectbox(
        "TTS 언어 선택",
        options=[("한국어 (ko)", "ko"), ("영어 (en)", "en"), ("일본어 (ja)", "ja")],
        format_func=lambda x: x[0], # Display only the name in the selectbox
        index=0,
        help="예측된 한글 문자를 읽어줄 언어를 선택하세요."
    )
    tts_lang = tts_lang_option[1] # Get the language code

    st.sidebar.markdown("---")
    st.sidebar.info("💡 이 앱은 Arduino(ESP-01)로부터 TCP 통신을 통해 점자 데이터를 수신합니다. 포트: 5001")

    # --- Auto Refresh (0.5 seconds interval) ---
    st_autorefresh(interval=500, limit=None, key="auto_refresh_braille")

    # ===== Main Content Layout: Two Columns (will stack on mobile) =====
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 1) 실시간 데이터")
        # Placeholder for the grid image
        grid_placeholder = st.empty()
        st.markdown(
            "<p style='text-align: center; color: #777; font-size: 0.9rem;'>8×10 센서 배열에서 읽어들인 점자 패턴입니다.</p>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### 2) 번역 결과")
        # Placeholder for the prediction result
        prediction_placeholder = st.empty()
        st.markdown(
            "<p style='text-align: center; color: #777; font-size: 0.9rem;'>AI 모델이 인식한 한글 문자와 음성 출력입니다.</p>",
            unsafe_allow_html=True
        )

    # ===== Model Inference & UI Update (runs on each refresh) =====
    tensor = None
    with lock:
        if latest_grid is not None:
            tensor = latest_grid.clone()

    if tensor is not None:
        # (1) 8x10 Image Reconstruction
        arr_8x10 = (tensor.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Update grid image in its placeholder
        with grid_placeholder.container():
            st.image(
                arr_8x10,
                caption="실시간 데이터",
                width=300, # Fixed width, responsive via CSS max-width
                use_column_width=False,
                clamp=True,
            )

        # (2) Model Inference
        if model is not None:
            with torch.no_grad():
                input_tensor = tensor.to(device)
                outputs = model(input_tensor)
                _, pred_idx = torch.max(outputs, dim=1)
                class_idx = pred_idx.item()

            class_names = load_class_names()
            predicted_char = class_names[class_idx] if 0 <= class_idx < len(class_names) else "알 수 없음"

            # (3) Generate TTS only if it's a new prediction
            if predicted_char != latest_prediction:
                latest_prediction = predicted_char
                if predicted_char != "알 수 없음":
                    try:
                        tts = gTTS(text=predicted_char, lang=tts_lang)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
                            tts.save(tmp_audio_file.name)
                            tmp_path = tmp_audio_file.name
                        with open(tmp_path, "rb") as f:
                            audio_bytes = f.read()
                        os.unlink(tmp_path) # Clean up temp file
                    except Exception as e:
                        audio_bytes = None
                        # print(f"TTS generation error: {e}") # For debugging
                else:
                    audio_bytes = None # No audio for "Unknown"

            # Update prediction result in its placeholder
            with prediction_placeholder.container():
                st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
                st.markdown(f"<h2>{predicted_char}</h2>", unsafe_allow_html=True)
                st.markdown("<p>번역 결과</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", help="예측된 문자의 음성 출력입니다.")
                elif predicted_char == "알 수 없음":
                    st.warning("모델이 점자를 정확히 인식하지 못했습니다. 더 많은 데이터가 필요할 수 있습니다.")
                else:
                    st.info("음성 파일을 생성 중입니다...")
        else:
            with prediction_placeholder.container():
                st.warning("모델이 로드되지 않았습니다. 사이드바에서 모델을 로드해주세요.")

    else:
        # Initial state or no data received
        with grid_placeholder.container():
            st.info("🔄 Arduino(ESP-01)로부터 데이터 수신 대기 중입니다. 잠시만 기다려주세요.")
        with prediction_placeholder.container():
            st.info("예측 결과가 여기에 표시될 예정입니다. 센서 데이터를 기다리고 있습니다.")


# ------------------------------------------------
# 6. 백그라운드 스레드 실행 & Streamlit 앱 시작
# ------------------------------------------------
if __name__ == "__main__":
    # Ensure the socket listener only starts once
    if "listener_started" not in st.session_state:
        listener_thread = threading.Thread(target=socket_listener, daemon=True)
        listener_thread.start()
        st.session_state.listener_started = True

    # Streamlit app execution
    main()