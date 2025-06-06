import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from gtts import gTTS
import tempfile
import os
from streamlit_autorefresh import st_autorefresh # pip install streamlit-autorefresh
import serial
import time

from css import css_string
from utils import KOREAN_BRAILLE_MAP
from BrailleToKorean.BrailleToKor import BrailleToKor

import base64

# --- PyTorch Model Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {DEVICE}")

TRANSLATOR = BrailleToKor()

# ------------------------------------------------
# 1. Braille Classification Model (Updated from Notebook)
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
# Helper Functions
# ------------------------------------------------
# IMPORTANT: Ensure 'braille_cnn_model.pth' matches the BrailleCNN architecture defined above.
# If using a model from the notebook saved as 'braille_recognition_model.pth',
# ensure it was trained with the same architecture (16ch -> 32ch conv).
def load_braille_model(model_path="braille_cnn_model.pth"): # Default path, consider changing if notebook model is primary
    print(f"[DEBUG] Attempting to load model from: {model_path}")
    try:
        # Pass grid_rows and grid_cols if your BrailleCNN __init__ needs them
        # For this example, assuming 8x10 fixed for fc layer calculation
        model = BrailleCNN()
        if not os.path.exists(model_path):
            st.sidebar.error(f"⚠️ 모델 파일 '{model_path}'을 찾을 수 없습니다. 경로를 확인하거나 모델을 학습시켜주세요.")
            print(f"[DEBUG] Model file not found: {model_path}")
            return None

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.sidebar.success("✅ 모델이 성공적으로 로드되었습니다!")
        print("[DEBUG] Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.sidebar.error(f"⚠️ '{model_path}' 파일을 찾을 수 없습니다. 모델 파일 위치를 확인하세요.")
        print(f"[DEBUG] Model file not found (double check): {model_path}")
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ 모델 로드 중 오류 발생: {e}")
        print(f"[DEBUG] Error loading model: {e}")
        return None

def map_6_dots_to_character(dot_pattern):
    """Maps a 6-dot binary pattern (tuple) to a Korean character."""
    # Ensure dot_pattern is a tuple for dictionary key lookup
    if isinstance(dot_pattern, np.ndarray):
        dot_pattern = tuple(dot_pattern.astype(int).tolist())
    elif isinstance(dot_pattern, list):
        dot_pattern = tuple(dot_pattern)

    char = KOREAN_BRAILLE_MAP.get(dot_pattern, "알 수 없음")
    # print(f"[DEBUG] Mapping dot pattern {dot_pattern} to char '{char}'") # Can be verbose
    return char


def read_and_process_serial_data(serial_conn, min_sensor_val=0.0, max_sensor_val=1024.0, target_rows=10, target_cols=8):
    grid_list_of_lists = []
    total_values_expected = target_rows * target_cols
    values_read = 0
    
    # print(f"[DEBUG] Attempting to read {target_rows}x{target_cols} grid ({total_values_expected} values).")

    if not (serial_conn and serial_conn.is_open):
        # print("[DEBUG] Serial connection not open in read_and_process_serial_data.")
        return None

    current_line_buffer = "" # Buffer for incomplete lines if any

    for r in range(target_rows):
        current_row = []
        for c in range(target_cols):
            value_found_for_cell = False
            while not value_found_for_cell:
                if serial_conn.in_waiting > 0:
                    try:
                        # Readline might block until newline or timeout
                        line_bytes = serial_conn.readline()
                        line = line_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if line: # Ensure line is not empty
                            value = int(line)
                            print(f"[SENSOR RAW] ({r},{c}): {value}") # 원시 센서 값 로깅 (디버깅용)
                            current_row.append(value)
                            values_read += 1
                            value_found_for_cell = True
                            # print(f"[DEBUG] Read {value} for cell ({r},{c})")
                            break # Move to next cell
                        # else: empty line, loop again if within timeout
                            # print(f"[DEBUG] Empty line at ({r},{c}), waiting...")
                    except ValueError:
                        print(f"[DEBUG] ValueError at ({r},{c}). Line: '{line}'")
                        st.toast(f"⚠️ 데이터 형식 오류: '{line[:10]}...'", icon="❗")
                        return None
                    except Exception as e:
                        st.toast(f"⚠️ 데이터 읽기 오류 ({r},{c}): {e}", icon="�")
                        print(f"[DEBUG] Unexpected error reading serial data ({r},{c}): {e}")
                        return None
                else: # No data in waiting, sleep briefly before retrying within cell timeout
                    time.sleep(0.01) # Small sleep to prevent busy-waiting
            
            if not value_found_for_cell:
                print(f"[DEBUG] Timeout or no data for cell ({r},{c}) after attempts.")
                st.toast(f"⚠️ 셀 ({r},{c}) 데이터 수신 시간 초과", icon="⏱️")
                return None # Failed to get data for this cell

        grid_list_of_lists.append(current_row)

    if len(grid_list_of_lists) == target_rows and all(len(r) == target_cols for r in grid_list_of_lists):
        print(f"[DEBUG] Successfully read {values_read}/{total_values_expected} values.")
        input = np.array(grid_list_of_lists, dtype=np.float32)
        input = np.clip(input, min_sensor_val, max_sensor_val)
        
        min_val = np.min(input)
        max_val = np.max(input)
        normalized_input = (input - min_val) / (max_val - min_val)
        normalized_input = np.expand_dims(normalized_input, axis=0)
        normalized_input = np.expand_dims(normalized_input, axis=0)

        input_tensor = torch.from_numpy(normalized_input).float()

        return input_tensor
    else:
        print(f"[DEBUG] Incomplete grid. Read {values_read}/{total_values_expected}. Structure: {len(grid_list_of_lists)} rows.")
        return None

# ------------------------------------------------
# Streamlit main()
# ------------------------------------------------
def main():
    print("\n[DEBUG] --- Streamlit Script Rerun ---")

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'latest_grid' not in st.session_state:
        st.session_state.latest_grid = None
    if 'latest_predicted_dots' not in st.session_state: # Store the 6-dot pattern
        st.session_state.latest_predicted_dots = None
    if 'latest_prediction_char' not in st.session_state:
        st.session_state.latest_prediction_char = None
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    if 'serial_connection' not in st.session_state:
        st.session_state.serial_connection = None
    if 'serial_is_running' not in st.session_state:
        st.session_state.serial_is_running = False
    if 'serial_port_input' not in st.session_state:
        st.session_state.serial_port_input = 'COM9' # Default, adjust as needed
    if 'serial_baud_rate_input' not in st.session_state:
        st.session_state.serial_baud_rate_input = 9600
    if 'auto_connect_attempted_this_session' not in st.session_state:
        st.session_state.auto_connect_attempted_this_session = False
    if 'current_tts_lang' not in st.session_state:
        st.session_state.current_tts_lang = "ko"
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    if 'translated_sentence' not in st.session_state:
        st.session_state.translated_sentence = ""
        

    st.set_page_config(page_title="O-MOK", layout="wide", initial_sidebar_state="auto")

    if not st.session_state.auto_connect_attempted_this_session and \
       not st.session_state.serial_is_running:
        print("[DEBUG] Attempting automatic serial connection on first load...")
        port_to_try = st.session_state.serial_port_input
        baud_to_try = st.session_state.serial_baud_rate_input
        try:
            if st.session_state.serial_connection and st.session_state.serial_connection.is_open:
                st.session_state.serial_connection.close()
            st.session_state.serial_connection = serial.Serial(port_to_try, baud_to_try, timeout=0.1) # Shorter timeout for connect
            st.session_state.serial_is_running = True
            st.toast(f"✅ 자동 연결 성공: {port_to_try}", icon="🔌")
        except Exception as e:
            print(f"[DEBUG] Auto-connect FAILED for {port_to_try}: {e}")
            st.session_state.serial_connection = None
            st.session_state.serial_is_running = False
            st.toast(f"⚠️ 자동 연결 실패: {port_to_try}. ({e})", icon="🚫")
        finally:
            st.session_state.auto_connect_attempted_this_session = True

    st.markdown(css_string, unsafe_allow_html=True)
    st.title("👁️‍🗨️ O-MOK")
    st.markdown("<p class='intro-text'>손 끝에 정보를 담는 <b>실시간 점자 번역기</b>입니다.<br>실시간 점자 데이터를 <b>AI 모델</b>이 즉시 분류하고,<br>인식된 한글 문자를 <b>음성</b>으로 들려줍니다.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.header("⚙️ 앱 설정")
    st.sidebar.markdown("### 🔌 시리얼 연결")
    current_port = st.session_state.serial_port_input
    current_baud = st.session_state.serial_baud_rate_input
    st.session_state.serial_port_input = st.sidebar.text_input("시리얼 포트", value=current_port)
    st.session_state.serial_baud_rate_input = st.sidebar.selectbox("보드 레이트", options=[9600, 115200, 57600, 38400], index=([9600, 115200, 57600, 38400].index(current_baud) if current_baud in [9600, 115200, 57600, 38400] else 0))

    col_connect, col_disconnect = st.sidebar.columns(2)
    if col_connect.button("연결 시작", disabled=st.session_state.serial_is_running, use_container_width=True):
        if st.session_state.serial_connection and st.session_state.serial_connection.is_open:
            st.session_state.serial_connection.close()
        try:
            st.session_state.serial_connection = serial.Serial(st.session_state.serial_port_input, st.session_state.serial_baud_rate_input, timeout=0.1) # Shorter timeout
            st.session_state.serial_is_running = True
            st.sidebar.success(f"✅ {st.session_state.serial_port_input}에 연결됨.")
            st.session_state.latest_grid = None; st.session_state.latest_prediction_char = None; st.session_state.audio_bytes = None
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"⚠️ 연결 실패: {e}")
            st.session_state.serial_connection = None; st.session_state.serial_is_running = False

    if col_disconnect.button("연결 중지", disabled=not st.session_state.serial_is_running, use_container_width=True):
        st.session_state.serial_is_running = False
        if st.session_state.serial_connection and st.session_state.serial_connection.is_open:
            st.session_state.serial_connection.close()
            st.sidebar.info("🔌 연결이 중지되었습니다.")
        st.rerun()

    st.sidebar.markdown("### 🧠 모델 관리")
    # Ensure model path matches your actual model file, possibly 'braille_recognition_model.pth' from notebook
    model_file_to_load = "braille_cnn_model.pth" # Or "braille_recognition_model.pth"
    if st.sidebar.button("모델 (재)로드"):
        with st.spinner("모델을 다시 로드 중..."):
            st.session_state.model = load_braille_model(model_file_to_load)
    if st.session_state.model is None:
        with st.spinner("초기 모델 로드 중..."):
            st.session_state.model = load_braille_model(model_file_to_load)

    st.sidebar.markdown("### 🔊 음성 합성 설정")
    tts_lang_options_map = {"한국어 (ko)": "ko", "영어 (en)": "en", "일본어 (ja)": "ja"}
    selected_tts_lang_display = st.sidebar.selectbox("TTS 언어 선택", options=list(tts_lang_options_map.keys()), index=list(tts_lang_options_map.values()).index(st.session_state.current_tts_lang))
    st.session_state.current_tts_lang = tts_lang_options_map[selected_tts_lang_display]
    st.sidebar.markdown("---")

    st_autorefresh(interval=10000, limit=None, key="auto_refresh_braille_serial") # Slightly longer interval

    if st.session_state.serial_is_running and st.session_state.serial_connection:
        # Heuristic: check if there's roughly enough data for a full grid
        # (80 values * ~2 bytes/value for "0\n" or "1\n")
        MIN_BYTES_FOR_GRID = 6 * 80 # Minimum if just '0' or '1' without newline, adjust as needed
        print(f"[DEBUG] in_waiting = {st.session_state.serial_connection.in_waiting}")
        if st.session_state.serial_connection.in_waiting >= MIN_BYTES_FOR_GRID:
            print(f"[DEBUG] Buffer: {st.session_state.serial_connection.in_waiting} bytes. Attempting grid read.")
            # This read_and_process_serial_data will now try to read 80 individual lines
            new_grid_tensor = read_and_process_serial_data(st.session_state.serial_connection, target_rows=10, target_cols=8)
            print(f"[DEBUG] Returned tensor shape: {new_grid_tensor.shape}")
            print(f"[DEBUG] Type of new_grid_tensor: {type(new_grid_tensor)}, device: {new_grid_tensor.device}")
            st.session_state.latest_grid = new_grid_tensor
            print(f"[DEBUG] Update latest_grid")

            with torch.no_grad():
                input_for_model = st.session_state.latest_grid.to(DEVICE)
                print(f"[DEBUG] 모델 추론 시작: {input_for_model.shape}")
                model_outputs = st.session_state.model(input_for_model)
                # --- Output processing for multi-label (6-dot) ---
                predicted_labels = (torch.sigmoid(model_outputs) > 0.5).int().cpu().numpy().flatten().tolist()
                print(f"[DEBUG] 모델 추론 결과: {predicted_labels}")
                
                st.session_state.sentence += KOREAN_BRAILLE_MAP.get(str(predicted_labels))
                print(f"[DEBUG] RAW BRAILLE: '{st.session_state.sentence}'")
                st.session_state.translated_sentence = TRANSLATOR.translation(st.session_state.sentence)
                print(f"[DEBUG] 번역 결과: '{st.session_state.translated_sentence}'")

                if st.session_state.translated_sentence:
                    tts = gTTS(text="번역 결과는 다음과 같습니다. " + st.session_state.translated_sentence, lang="ko")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tts.save(tmp.name)
                        tmp_path = tmp.name
                    
                    with open(tmp_path, "rb") as f:
                        st.session_state.audio_bytes = f.read()
                        print(f"[DEBUG] audio_bytes length = {len(st.session_state.audio_bytes)} bytes")
                    os.unlink(tmp_path)

                    b64 = base64.b64encode(st.session_state.audio_bytes).decode("utf-8")

                    # (C) HTML <audio> 태그 작성
                    audio_html = f"""
                    <audio controls autoplay preload="auto">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3" />
                    Your browser does not support the audio element.
                    </audio>
                    """

                    # (D) Streamlit에 렌더
                    st.markdown(audio_html, unsafe_allow_html=True)
                    # Prediction logic moved down to ensure it runs after potential grid update
            # else:
                # print("[DEBUG] read_and_process_serial_data returned None.")
        # else:
            # print(f"[DEBUG] Buffer low: {st.session_state.serial_connection.in_waiting} bytes. Waiting.")
            # pass # Wait for more data

    elif st.session_state.serial_is_running and not st.session_state.serial_connection:
        st.sidebar.warning("⚠️ 연결 상태 오류. 재연결 필요.")
        st.session_state.serial_is_running = False

    col_display1, col_display2 = st.columns([1, 1])
    with col_display1:
        st.markdown("### 1) 실시간 데이터")
        grid_placeholder = st.empty()
        st.markdown("<p style='text-align: center; color: #777;'>10×8 센서 배열</p>", unsafe_allow_html=True)
    with col_display2:
        st.markdown("### 2) 번역 결과")
        prediction_placeholder = st.empty()
        st.markdown("<p style='text-align: center; color: #777;'>AI 인식 결과</p>", unsafe_allow_html=True)

    if st.session_state.latest_grid is not None:
        with grid_placeholder.container():
            arr_display = (st.session_state.latest_grid.squeeze().cpu().numpy() * 255).astype(np.uint8)
            st.image(arr_display, caption="실시간 데이터 (10x8)", width=300, use_container_width='auto', clamp=True)

        if st.session_state.model is not None:
            with prediction_placeholder.container():
                st.markdown(f"<div class='prediction-container'><h2>{st.session_state.translated_sentence if st.session_state.translated_sentence else '...'}</h2><p>번역 결과</p></div>", unsafe_allow_html=True)
        else:
            with prediction_placeholder.container(): st.warning("⚠️ 모델 로드 안됨.")
    else:
        with grid_placeholder.container():
            st.info("🔌 시리얼 연결 시작 또는 데이터 수신 대기 중..." if st.session_state.serial_is_running else "🔌 시리얼 연결을 시작해주세요.")
        with prediction_placeholder.container(): st.info("👀 결과 대기 중...")

if __name__ == "__main__":
    main()