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
from streamlit_autorefresh import st_autorefresh

latest_grid = None           
latest_prediction = None     
audio_bytes = None           

model = None                 
device = torch.device("cpu") 
lock = threading.Lock()      

# ------------------------------------------------
# 1. 브라유 인식용 PyTorch 네트워크 클래스 정의
#    (모델 로드할 때 동일 클래스가 필요합니다)
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
        # 입력 (N, 1, 8, 10) -> conv1 (N, 32, 8, 10) -> pool1 (N, 32, 4, 5)
        # -> conv2 (N, 64, 4, 5) -> pool2 (N, 64, 2, 2) [5//2=2]
        # 평탄화 후 특징 수: 64 * 2 * 2 = 256
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6) # 6개 점자 예측

    def forward(self, x):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1) # 평탄화
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
    srv_sock.bind((host, port))
    srv_sock.listen(1)  # 한 번에 하나의 Arduino(ESP-01) 연결만 허용

    while True:
        conn, addr = srv_sock.accept()
        data_bytes = b''
        # Arduino가 보낸 JSON을 chunk 단위로 모두 받아온다
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data_bytes += chunk
        conn.close()

        try:
            payload = json.loads(data_bytes.decode('utf-8'))
            grid_list = payload.get("grid", None)
            if grid_list is not None:
                arr = np.array(grid_list, dtype=np.float32)  # (8,10) 형태
                if arr.shape == (8, 10):
                    # PyTorch 용 텐서: (1,1,8,10)
                    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                    # 정규화: 0~255 값을 0~1로 스케일링
                    tensor = tensor / 255.0
                    with lock:
                        latest_grid = tensor.clone()
        except Exception:
            # JSON 파싱 오류나 shape 오류 무시
            pass

        time.sleep(0.05)  # 과도한 CPU 사용 방지

# ------------------------------------------------
# 4. 클래스 이름(인덱스 ↔ 한글 음절) 매핑 함수
#    - 모델 학습 시 사용된 순서와 **완전히 동일하게** 유지해야 함
# ------------------------------------------------
@st.cache_resource
def load_class_names():
    # 예시: 총 40개의 브라유 한글 음절을 학습했다고 가정
    # **절대로 순서를 바꾸지 마세요!**
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
        page_title="PyTorch 실시간 브라유 인식",
        layout="wide"
    )
    st.title("📱 Streamlit + PyTorch 브라유(Braille) 인식 데모")
    st.markdown(
        """
        - **Arduino Nano + ESP-01**에서 Wi-Fi(TCP)로 8×10 센서 데이터를 전송  
        - **PyTorch 모델**(`braille_model.pth`)을 로드하여 실시간 추론  
        - **gTTS**를 사용해 예측된 한글 음절을 MP3로 생성 → `st.audio()`로 브라우저/모바일에서 재생  
        """
    )

    # ===== 5-1. 사이드바: 모델 로드/재로드, TTS 언어 설정 =====
    st.sidebar.header("설정")
    if st.sidebar.button("모델 재로드"):
        with st.spinner("모델을 다시 로드 중..."):
            model = BrailleCNN()
            model.load_state_dict(torch.load("braille_cnn_model.pth", map_location=device))
            model.to(device)
            model.eval()
        st.sidebar.success("모델이 정상적으로 재로드되었습니다!")

    if model is None:
        # 첫 실행 시 모델 로드
        with st.spinner("초기 모델 로드 중..."):
            model = BrailleCNN()
            model.load_state_dict(torch.load("braille_cnn_model.pth", map_location=device))
            model.to(device)
            model.eval()
        st.sidebar.success("모델이 로드되었습니다!")

    # TTS 언어 (ko/en/ja 등)
    tts_lang = st.sidebar.selectbox(
        "TTS 언어 선택",
        options=["ko", "en", "ja"],
        index=0
    )

    # ===== 5-2. 자동 새로고침 설정 =====
    # 페이지가 로드될 때마다(또는 0.5초마다) 다시 실행됨
    # interval=500 → 500ms(0.5초)마다 새로고침
    count = st_autorefresh(interval=500, limit=None, key="auto_refresh")

    # ===== 5-3. UI 플레이스홀더 =====
    st.subheader("1) 마지막으로 수신된 8×10 센서 그리드")
    grid_placeholder = st.empty()

    st.subheader("2) 예측된 브라유 문자 → 한글 출력")
    prediction_placeholder = st.empty()

    # ===== 5-4. 최신 그리드, 모델 추론, TTS 생성 & 플레이 =====
    with lock:
        tensor = latest_grid.clone() if latest_grid is not None else None

    if tensor is not None:
        # (1,1,8,10) → (8,10) NumPy 배열 (0~255 흑백)로 복원
        arr_8x10 = (tensor.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        # 8×10 이미지로 시각화
        grid_placeholder.image(
            arr_8x10,
            caption="8×10 센서 그리드 (0~255 그레이스케일)",
            width=160,
            use_column_width=False,
            clamp=True,
        )

        # PyTorch 모델 추론
        with torch.no_grad():
            input_tensor = tensor.to(device)  # (1,1,8,10) float32, 0~1 범위
            outputs = model(input_tensor)     # (1, num_classes) logits
            _, pred_idx = torch.max(outputs, dim=1)
            class_idx = pred_idx.item()

        # 인덱스를 한글 음절로 변환
        class_names = load_class_names()
        if 0 <= class_idx < len(class_names):
            predicted_char = class_names[class_idx]
        else:
            predicted_char = "알 수 없음"

        # 이전 예측 결과와 다르면, gTTS로 음성 생성
        if predicted_char != latest_prediction:
            latest_prediction = predicted_char
            tts = gTTS(text=predicted_char, lang=tts_lang)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            tmp.seek(0)
            audio_bytes = tmp.read()

        # UI에 텍스트 결과 출력 & 음성 재생
        prediction_placeholder.markdown(
            f"**예측 결과:** {predicted_char}"
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    else:
        # 아직 Arduino 쪽에서 그리드를 한 번도 받지 못한 경우
        grid_placeholder.info("아직 Arduino(ESP-01)로부터 데이터가 수신되지 않았습니다.")
        prediction_placeholder.info("예측 결과가 없습니다.")


# ------------------------------------------------
# 6. 백그라운드 스레드 실행 & Streamlit 앱 시작
# ------------------------------------------------
if __name__ == "__main__":
    # 백그라운드에서 TCP 소켓 리스너 시작
    listener_thread = threading.Thread(target=socket_listener, daemon=True)
    listener_thread.start()

    # Streamlit 앱 실행
    main()
