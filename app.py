import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import requests

st.set_page_config(page_title="Dự đoán bệnh lá cà chua", layout="wide")
st.title("Chẩn đoán bệnh trên lá Cà Chua") 
st.write(
    "Tải lên ảnh lá cà chua hoặc nhập URL ảnh. Hệ thống sẽ phân tích và cung cấp thông tin chi tiết về bệnh (nếu có), "
    "bao gồm dấu hiệu nhận dạng, tên khoa học, và các biện pháp quản lý tham khảo."
)

# ÁNH XẠ TÊN LỚP (Giữ nguyên)
TURKISH_TO_ENGLISH_CLASS_MAP = {
    'Erken yaniklik': 'Early_blight',
    'Gec yaniklik': 'Late_blight',
    'Mozaik Virusu': 'Tomato_mosaic_virus',
    'Orumcek akarlari': 'Spider_mites Two-spotted_spider_mite',
    'Saglikli': 'healthy',
    'Sari Yaprak Kivrilma Virusu': 'Tomato_Yellow_Leaf_Curl_Virus',
    'Septorya': 'Septoria_leaf_spot',
    'Yaprak Kufu': 'Leaf_Mold',
    'Yaprak madencisi': 'Leaf_miner'
}

# THÔNG TIN BỆNH VÀ CÁCH XỬ LÝ
# *** QUAN TRỌNG: Bạn cần tự điền thông tin chi tiết và chính xác cho 'identification_signs' của TẤT CẢ các bệnh ***
disease_info = {
    'Bacterial_spot': {
        'scientific_name_en': "Xanthomonas spp.",
        'vietnamese_name': "Bệnh đốm khuẩn",
        'identification_signs': [
            "Đốm nhỏ, sũng nước, màu xanh đậm đến đen trên lá, thường có viền vàng.",
            "Các đốm có thể liên kết lại thành mảng lớn, làm lá bị rách hoặc biến dạng.",
            "Trên quả, vết bệnh nổi gờ, màu nâu đen, có vảy."
            # Thêm các dấu hiệu khác nếu có
        ],
        'remedies': ["Sử dụng thuốc trừ bệnh gốc Đồng (Copper Oxychloride, Copper Hydroxide).", "Xem xét kháng sinh chuyên dùng (Streptomycin, Kasugamycin) khi áp lực bệnh rất cao và tuân thủ nghiêm ngặt hướng dẫn."],
        'actions': ["Thu gom và tiêu hủy bộ phận cây bị bệnh.", "Luân canh cây trồng (tránh họ cà 2-3 năm).", "Chọn giống kháng bệnh.", "Tránh tưới lên lá, giữ lá khô ráo.", "Khử trùng dụng cụ làm vườn."]
    },
    'Early_blight': {
        'scientific_name_en': "Alternaria solani",
        'vietnamese_name': "Bệnh cháy sớm (Đốm vòng)",
        'identification_signs': [
            "Vết bệnh hình tròn hoặc góc cạnh, màu nâu sẫm, có các vòng tròn đồng tâm đặc trưng như 'bia bắn'.",
            "Thường xuất hiện ở các lá già phía dưới trước, sau đó lan dần lên trên.",
            "Lá bị bệnh nặng sẽ vàng, khô và rụng sớm.",
            "Trên thân và cuống lá có thể có vết bệnh hình bầu dục, màu nâu đen."
        ],
        'remedies': ["Phun thuốc trừ nấm chứa hoạt chất Mancozeb, Chlorothalonil khi bệnh mới xuất hiện.", "Các hoạt chất Azoxystrobin, Difenoconazole cũng hiệu quả; nên luân phiên thuốc."],
        'actions': ["Dọn sạch tàn dư cây trồng vụ trước.", "Đảm bảo ruộng thoát nước tốt, tránh úng ngập.", "Luân canh với cây trồng khác họ.", "Bón phân cân đối, tránh thừa đạm, tăng cường Kali và Canxi.", "Cắt tỉa lá già, lá bệnh ở gốc."]
    },
    'Late_blight': {
        'scientific_name_en': "Phytophthora infestans",
        'vietnamese_name': "Bệnh sương mai (Mốc sương)",
        'identification_signs': [
            "Trên lá xuất hiện các đốm màu xanh xám, úng nước, sau đó lớn dần và chuyển sang nâu đen.",
            "Ở mặt dưới lá, tại rìa vết bệnh, có thể thấy lớp mốc trắng xốp khi thời tiết ẩm ướt.",
            "Bệnh phát triển rất nhanh, có thể làm toàn bộ lá, thân cây bị thối nhũn và chết rũ.",
            "Trên quả, vết bệnh màu nâu, cứng, lan sâu vào thịt quả."
        ],
        'remedies': ["Phun thuốc phòng trừ chủ động khi thời tiết thuận lợi cho bệnh (ẩm, mưa nhiều, sương mù).", "Sử dụng thuốc có hoạt chất: Mancozeb + Metalaxyl, Cymoxanil + Mancozeb, Propamocarb, Dimethomorph.", "Luân phiên thuốc để tránh kháng."],
        'actions': ["Chọn giống kháng bệnh.", "Trồng với mật độ hợp lý, đảm bảo thông thoáng.", "Quản lý nước tốt, tránh đọng nước.", "Tiêu hủy kịp thời cây, lá bị bệnh nặng.", "Luân canh nghiêm ngặt."]
    },
    'Leaf_Mold': {
        'scientific_name_en': "Fulvia fulva (syn. Cladosporium fulvum)",
        'vietnamese_name': "Bệnh mốc lá",
        'identification_signs': [
            "Mặt trên lá xuất hiện các đốm màu vàng nhạt hoặc xanh nhạt, không rõ ràng.",
            "Mặt dưới lá, tương ứng với các đốm đó, là lớp nấm mốc màu xanh ôliu đến nâu nhạt, mịn như nhung.",
            "Lá bị bệnh nặng có thể cong lại, vàng và khô héo."
        ],
        'remedies': ["Sử dụng thuốc trừ nấm gốc Đồng, Chlorothalonil, Azoxystrobin, Trifloxystrobin.", "Phun kỹ mặt dưới lá."],
        'actions': ["Đảm bảo thông gió tốt, đặc biệt trong nhà kính/nhà lưới.", "Kiểm soát độ ẩm không khí, tránh tưới chiều tối.", "Cắt tỉa lá già, lá gốc và lá bệnh.", "Vệ sinh tàn dư thực vật."]
    },
    'Septoria_leaf_spot': {
        'scientific_name_en': "Septoria lycopersici",
        'vietnamese_name': "Bệnh đốm lá Septoria",
        'identification_signs': [
            "Đốm bệnh nhỏ, tròn, màu nâu xám hoặc nâu nhạt, có tâm màu trắng hoặc xám tro.",
            "Trong các đốm bệnh già có thể thấy các chấm đen nhỏ li ti (bào tử của nấm).",
            "Bệnh thường bắt đầu từ lá dưới và lan dần lên, làm lá vàng, khô và rụng hàng loạt."
        ],
        'remedies': ["Phun thuốc trừ nấm chứa Chlorothalonil, Mancozeb.", "Thuốc gốc Đồng có tác dụng phòng trừ."],
        'actions': ["Thu gom và tiêu hủy lá bệnh.", "Giữ vườn sạch sẽ.", "Luân canh ít nhất 1-2 năm.", "Ưu tiên tưới gốc.", "Bón phân cân đối, tăng cường hữu cơ và kali."]
    },
    'Spider_mites Two-spotted_spider_mite': {
        'scientific_name_en': "Tetranychus urticae",
        'vietnamese_name': "Nhện đỏ hai chấm",
        'identification_signs': [
            "Lá bị hại có những chấm nhỏ li ti màu vàng hoặc trắng bạc do nhện chích hút dịch.",
            "Mặt dưới lá có thể thấy tơ nhện mỏng và các con nhện nhỏ li ti (cần kính lúp để thấy rõ).",
            "Lá bị nặng có thể chuyển vàng, khô và rụng. Ngọn cây có thể bị chùn lại."
        ],
        'remedies': ["Thuốc đặc trị nhện: Abamectin, Emamectin Benzoate, Spiromesifen, Hexythiazox.", "Sản phẩm sinh học: dầu khoáng, nấm Beauveria bassiana.", "Phun kỹ mặt dưới lá, lặp lại nếu cần."],
        'actions': ["Phun nước mạnh vào mặt dưới lá (khi mật độ thấp).", "Bảo tồn thiên địch (bọ rùa, nhện bắt mồi).", "Cắt tỉa và tiêu hủy lá, cành bị hại nặng.", "Duy trì độ ẩm thích hợp (nhện phát triển mạnh khi khô nóng)."]
    },
    'Target_Spot': {
        'scientific_name_en': "Corynespora cassiicola",
        'vietnamese_name': "Bệnh đốm mắt cua",
        'identification_signs': [
            "Vết bệnh trên lá có hình tròn hoặc không đều, màu nâu, thường có các vòng đồng tâm nhưng không rõ như bệnh cháy sớm.",
            "Tâm vết bệnh có thể bị thủng.",
            "Trên quả, vết bệnh lõm xuống, màu nâu sẫm."
            # Thêm các dấu hiệu khác
        ],
        'remedies': ["Sử dụng thuốc trừ nấm có hoạt chất Chlorothalonil, Mancozeb.", "Các thuốc nhóm Strobilurin (ví dụ: Azoxystrobin, Pyraclostrobin) cũng cho thấy hiệu quả tốt."],
        'actions': ["Thu dọn và tiêu hủy tàn dư cây bệnh từ vụ trước.", "Đảm bảo vườn trồng thông thoáng, tránh ẩm độ cao kéo dài.", "Luân canh với các cây trồng không phải là ký chủ của nấm.", "Bón phân cân đối, không bón thừa đạm."]
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'scientific_name_en': "Tomato yellow leaf curl virus (TYLCV)",
        'vietnamese_name': "Virus xoăn vàng lá cà chua",
        'identification_signs': [
            "Lá non bị xoăn lại, vàng, mép lá cong lên trên hoặc vào trong.",
            "Cây sinh trưởng còi cọc, lóng thân ngắn lại, lá nhỏ hơn bình thường.",
            "Hoa có thể bị rụng, khả năng đậu quả kém, quả nhỏ và biến dạng."
        ],
        'remedies': ["**Không có thuốc đặc trị bệnh virus.**", "Kiểm soát bọ phấn trắng (môi giới): Imidacloprid, Thiamethoxam, Pymetrozin (luân phiên).", "Dầu khoáng, xà phòng côn trùng để giảm mật độ bọ phấn."],
        'actions': ["Nhổ bỏ và tiêu hủy ngay cây có triệu chứng.", "Sử dụng giống cà chua kháng hoặc chống chịu virus TYLCV.", "Diệt trừ cỏ dại (nơi trú ẩn của bọ phấn).", "Luân canh.", "Sử dụng nhà lưới mắt nhỏ ngăn bọ phấn, đặc biệt giai đoạn cây con."]
    },
    'Tomato_mosaic_virus': {
        'scientific_name_en': "Tomato mosaic virus (ToMV)",
        'vietnamese_name': "Virus khảm lá cà chua",
        'identification_signs': [
            "Lá có những mảng màu xanh đậm xen kẽ với mảng màu xanh nhạt hoặc vàng (khảm mosaic).",
            "Lá có thể bị biến dạng, nhăn nheo, kích thước nhỏ lại.",
            "Cây sinh trưởng kém, năng suất giảm."
        ],
        'remedies': ["**Không có thuốc đặc trị bệnh virus.** Tập trung phòng ngừa lây nhiễm."],
        'actions': ["Nhổ bỏ và tiêu hủy cây bệnh.", "Sử dụng giống kháng.", "Khử trùng dụng cụ làm việc thường xuyên.", "Hạn chế gây vết thương cơ giới cho cây.", "Người làm vườn không hút thuốc lá khi làm việc với cà chua (virus có thể tồn tại trong thuốc lá)."]
    },
    'healthy': {
        'scientific_name_en': "N/A (Healthy Plant)",
        'vietnamese_name': "Cây khỏe mạnh",
        'identification_signs': ["Lá xanh tốt, không có đốm bệnh, không biến dạng.", "Cây sinh trưởng bình thường, phát triển cân đối."],
        'remedies': ["Không cần xử lý thuốc bệnh. Tiếp tục duy trì các biện pháp chăm sóc tốt."],
        'actions': ["Tưới nước đủ ẩm.", "Bón phân cân đối và đầy đủ.", "Thăm vườn thường xuyên để phát hiện sớm sâu bệnh.", "Áp dụng các biện pháp phòng ngừa chung (vệ sinh, luân canh, giống tốt)."]
    },
    'Leaf_miner': {
        'scientific_name_en': "Liriomyza spp.",
        'vietnamese_name': "Sâu vẽ bùa / Ruồi đục lá",
        'identification_signs': [
            "Trên lá xuất hiện các đường ngoằn ngoèo màu trắng bạc hoặc xám tro do ấu trùng (dòi) ăn phá biểu bì lá.",
            "Đầu đường hầm có thể thấy chấm đen nhỏ (phân của sâu non).",
            "Lá bị hại nặng có thể giảm khả năng quang hợp, vàng úa và rụng."
        ],
        'remedies': ["Sử dụng thuốc có hoạt chất Abamectin, Cyromazine, Spinetoram khi mật độ sâu cao.", "Dầu khoáng hoặc dầu neem.", "Đặt bẫy dính màu vàng để bắt ruồi trưởng thành."],
        'actions': ["Ngắt bỏ và tiêu hủy lá bị sâu vẽ bùa nặng.", "Vệ sinh đồng ruộng, dọn sạch cỏ dại.", "Luân canh.", "Bảo vệ thiên địch (ong ký sinh)."]
    }
}

# THAM SỐ CẤU HÌNH (Giữ nguyên)
MODEL_PATH = "best.pt"
MODEL_CONFIDENCE_THRESHOLD = 0.25
UI_CONFIDENCE_THRESHOLD = 0.60
GREEN_RATIO_THRESHOLD = 0.30

# LOAD MODEL (Giữ nguyên)
@st.cache_resource(show_spinner="Đang tải model nhận dạng...")
def load_yolo_model(model_path):
    try:
        model_obj = YOLO(model_path)
        if not hasattr(model_obj, 'names') or not isinstance(model_obj.names, (list, dict)):
             st.error("Lỗi: Model YOLO không có thuộc tính 'names' hợp lệ.")
             return None
        return model_obj
    except Exception as e:
        st.error(f"Lỗi khi tải model YOLO: {e}")
        return None
model = load_yolo_model(MODEL_PATH)

# HÀM XỬ LÝ (Giữ nguyên)
def check_green_ratio(image: Image.Image):
    img_array = np.array(image.convert('RGB'))
    green_pixels = np.sum((img_array[:,:,1] > img_array[:,:,0]) & \
                          (img_array[:,:,1] > img_array[:,:,2]) & \
                          (img_array[:,:,1] > 30))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    ratio = green_pixels / total_pixels if total_pixels > 0 else 0
    return ratio

def predict_and_analyze(image_input: Image.Image):
    if model is None or not hasattr(model, 'names'):
        return 'MODEL_ERROR', "Model chưa được tải hoặc không hợp lệ.", 0.0
    green_pixel_ratio = check_green_ratio(image_input)
    if green_pixel_ratio < GREEN_RATIO_THRESHOLD:
        return 'LOW_GREEN', f"{green_pixel_ratio*100:.1f}%", 0.0
    try:
        results = model.predict(image_input, verbose=False, conf=MODEL_CONFIDENCE_THRESHOLD)
        if results and results[0].boxes.shape[0] > 0:
            boxes = results[0].boxes.cpu().numpy()
            highest_conf_idx = np.argmax(boxes.conf)
            predicted_class_index = int(boxes.cls[highest_conf_idx])
            confidence = float(boxes.conf[highest_conf_idx])
            turkish_class_name = ""
            if isinstance(model.names, dict):
                turkish_class_name = model.names.get(predicted_class_index)
            elif isinstance(model.names, list):
                 if 0 <= predicted_class_index < len(model.names):
                    turkish_class_name = model.names[predicted_class_index]
            if not turkish_class_name:
                 return 'PREDICTION_ERROR', f"Không thể lấy tên lớp cho index {predicted_class_index}.", 0.0
            predicted_class_key_for_disease_info = TURKISH_TO_ENGLISH_CLASS_MAP.get(turkish_class_name)
            if predicted_class_key_for_disease_info is None:
                return 'CLASS_KEY_MISMATCH', f"Lớp '{turkish_class_name}' (từ model) không có trong ánh xạ.", confidence
            if predicted_class_key_for_disease_info not in disease_info:
                 return 'CLASS_KEY_MISMATCH', f"Thông tin cho '{predicted_class_key_for_disease_info}' (sau ánh xạ) không tồn tại.", confidence
            return 'OK', predicted_class_key_for_disease_info, confidence
        else:
            return 'NO_DETECTION', "Không phát hiện bệnh nào rõ ràng.", 0.0
    except Exception as e:
        return 'PREDICTION_ERROR', f"Lỗi xử lý model: {str(e)}", 0.0

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi tải ảnh từ URL: {e}")
        return None
    except IOError:
        st.error("Lỗi: URL không trỏ đến file ảnh hợp lệ hoặc định dạng ảnh không được hỗ trợ.")
        return None

# GIAO DIỆN
if model is None:
    st.error("Không thể tải Model YOLOv8. Ứng dụng không thể hoạt động.")
else:
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = {"status": None, "class_key": None, "confidence": None, "error_detail": None, "input_id": None}

    col1, col2 = st.columns([1, 2])

    with col1: # Cột nhập liệu (Giữ nguyên)
        st.header("Chọn Ảnh")
        input_method = st.radio(
            "Chọn phương thức nhập ảnh:",
            ('Tải ảnh lên', 'Nhập URL ảnh'),
            key='input_method_radio',
            horizontal=True # Hiển thị radio button nằm ngang cho gọn
        )
        image_to_analyze = None
        current_input_id = None
        if input_method == 'Tải ảnh lên':
            uploaded_file = st.file_uploader("Tải ảnh lá cà chua tại đây:", type=["jpg", "jpeg", "png"], key="file_uploader_key")
            if uploaded_file is not None:
                current_input_id = uploaded_file.name + str(uploaded_file.size)
                if st.session_state.last_analysis["input_id"] != current_input_id:
                    try: image_to_analyze = Image.open(uploaded_file)
                    except Exception as e:
                        st.error(f"Lỗi khi mở file ảnh: {e}")
                        st.session_state.last_analysis = {"status": "IMAGE_ERROR", "error_detail": str(e), "input_id": current_input_id}
                else:
                    try: image_to_analyze = Image.open(uploaded_file)
                    except: pass
                if image_to_analyze: st.image(image_to_analyze, caption="Ảnh đã chọn", use_container_width=True)
        elif input_method == 'Nhập URL ảnh':
            image_url = st.text_input("Nhập URL của ảnh lá cà chua:", key="image_url_input", placeholder="https://example.com/image.jpg")
            if st.button("Phân tích từ URL", key="analyze_url_button", use_container_width=True): # Nút rộng hơn
                if image_url:
                    current_input_id = image_url
                    # Luôn phân tích lại khi nhấn nút, không cần so sánh input_id ở đây nữa
                    with st.spinner("Đang tải và phân tích ảnh từ URL..."):
                        image_to_analyze = load_image_from_url(image_url)
                    if image_to_analyze: st.image(image_to_analyze, caption="Ảnh từ URL", use_container_width=True)
                else: st.warning("Vui lòng nhập URL ảnh.")
        
        trigger_analysis = False
        if image_to_analyze and current_input_id:
            if st.session_state.last_analysis["input_id"] != current_input_id:
                trigger_analysis = True
            # Nếu là URL và nút được nhấn, cũng trigger (đã xử lý ngầm khi image_to_analyze có giá trị từ nút URL)
            # Nếu là file upload, việc image_to_analyze có giá trị và input_id thay đổi là đủ trigger
        
        if trigger_analysis:
            with st.spinner("Đang phân tích ảnh..."):
                status, result_data, confidence_or_ratio = predict_and_analyze(image_to_analyze)
            if status == 'LOW_GREEN':
                st.session_state.last_analysis = {"status": status, "error_detail": result_data, "input_id": current_input_id}
            elif status == 'OK':
                st.session_state.last_analysis = {"status": status, "class_key": result_data, "confidence": confidence_or_ratio, "input_id": current_input_id}
            elif status in ['NO_DETECTION', 'CLASS_KEY_MISMATCH']:
                 st.session_state.last_analysis = {"status": status, "confidence": confidence_or_ratio if status == 'CLASS_KEY_MISMATCH' else None, "error_detail": result_data, "input_id": current_input_id}
            else: 
                st.session_state.last_analysis = {"status": status, "error_detail": result_data if result_data else "Lỗi không xác định.", "input_id": current_input_id}
        
        if input_method == 'Tải ảnh lên' and uploaded_file is None and st.session_state.last_analysis["input_id"] is not None:
             if not (st.session_state.last_analysis["input_id"] and st.session_state.last_analysis["input_id"].startswith("http")):
                st.session_state.last_analysis = {"status": None, "input_id": None}

    with col2: # Cột hiển thị kết quả
        st.header("Kết quả Phân Tích và Khuyến Nghị")
        analysis_result = st.session_state.last_analysis

        if analysis_result.get("status") == 'OK':
            predicted_class_key = analysis_result["class_key"]
            confidence = analysis_result["confidence"]
            display_confidence_percent = confidence * 100

            if predicted_class_key not in disease_info:
                st.error(f"Lỗi: Không tìm thấy thông tin cho bệnh '{predicted_class_key}'.")
            elif confidence <= UI_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"Độ tin cậy của dự đoán là **{display_confidence_percent:.2f}%**, "
                    f"không vượt qua ngưỡng **{UI_CONFIDENCE_THRESHOLD*100:.0f}%** để hiển thị chi tiết."
                )
                info_low_conf = disease_info[predicted_class_key]
                st.info(
                    f"Dựa trên dự đoán với độ tin cậy thấp, bệnh có thể là: **{info_low_conf['vietnamese_name']}** (Tên khoa học: *{info_low_conf['scientific_name_en']}*)."
                    "\n\nVui lòng thử lại với ảnh rõ nét hơn, hoặc tham khảo ý kiến chuyên gia để có chẩn đoán chính xác."
                )
            else: # Độ tin cậy > 60% và có thông tin bệnh
                info = disease_info[predicted_class_key]
                
                st.subheader(f"Chẩn đoán: {info['vietnamese_name']}")
                st.markdown(f"   - **Tên khoa học (Tiếng Anh):** *{info['scientific_name_en']}*")
                st.markdown(f"   - **Độ tin cậy của chẩn đoán:** **{display_confidence_percent:.2f}%**")

                # Hiển thị Dấu hiệu nhận dạng bệnh
                if info.get('identification_signs'):
                    with st.expander("**Dấu hiệu nhận dạng chính**", expanded=True):
                        for sign in info['identification_signs']:
                            st.write(f"• {sign}")
                else:
                    st.info("Hiện chưa có thông tin chi tiết về dấu hiệu nhận dạng cho bệnh này trong cơ sở dữ liệu.")

                if predicted_class_key != 'healthy':
                    st.markdown("---")
                    st.subheader("Khuyến Nghị Quản Lý Bệnh")
                    with st.expander("**Biện pháp hóa học (Tham khảo)**", expanded=True):
                        if info.get('remedies'):
                            for remedy in info['remedies']: st.write(f"• {remedy}")
                        else: st.write("• Không có gợi ý thuốc cụ thể cho trường hợp này.")
                    
                    with st.expander("**Biện pháp canh tác và phòng ngừa tổng hợp**", expanded=True):
                        if info.get('actions'):
                             for action in info['actions']: st.write(f"• {action}")
                        else: st.write("• Không có gợi ý biện pháp cụ thể.")
                    
                    st.warning( # Giữ nguyên Disclaimer quan trọng
                        """
                        **LƯU Ý QUAN TRỌNG VỀ THUỐC BVTV & BIỆN PHÁP CANH TÁC:**
                        \nThông tin gợi ý trên chỉ mang tính chất **tham khảo**. Hiệu quả thực tế phụ thuộc vào nhiều yếu tố (điều kiện thời tiết, giống cây, áp lực bệnh cụ thể, v.v.).
                        \n1. **Luôn đọc kỹ và tuân thủ tuyệt đối hướng dẫn sử dụng trên nhãn thuốc BVTV.**
                        \n2. Áp dụng nguyên tắc "4 đúng" (đúng thuốc, đúng lúc, đúng liều lượng & nồng độ, đúng cách) và đảm bảo thời gian cách ly.
                        \n3. **Hãy ưu tiên tham vấn ý kiến từ cán bộ kỹ thuật nông nghiệp hoặc chuyên gia bảo vệ thực vật tại địa phương của bạn để có giải pháp phù hợp, an toàn và hiệu quả nhất.**
                        \n4. Cân nhắc các biện pháp quản lý dịch hại tổng hợp (IPM) để giảm sự phụ thuộc vào hóa chất và bảo vệ môi trường.
                        """
                    )
                else: # Healthy và độ tin cậy > 60%
                    st.balloons()
                    st.success("**Cây cà chua của bạn được chẩn đoán là khỏe mạnh!**")
                    info_healthy = disease_info.get('healthy')
                    if info_healthy and info_healthy.get('actions'):
                        with st.expander("**Lời khuyên duy trì sức khỏe cho cây**", expanded=True):
                            for action in info_healthy['actions']:
                                st.write(f"• {action}")
                st.markdown("---")

        elif analysis_result.get("status") == 'LOW_GREEN':
            st.error(f"Ảnh có thể không phải là lá cà chua khả năng là lá cây: {analysis_result.get('error_detail','')}), vui lòng thử lại với ảnh khác hoặc cung cấp ảnh gần lá hơn.")
        elif analysis_result.get("status") == 'NO_DETECTION':
            st.warning(f"Không phát hiện đối tượng bệnh nào với độ tin cậy trên ngưỡng ({MODEL_CONFIDENCE_THRESHOLD*100:.0f}%). "
                       "Ảnh có thể là lá khỏe mạnh hoặc triệu chứng không rõ ràng.")
        elif analysis_result.get("status") == 'CLASS_KEY_MISMATCH':
            st.error(f"Lỗi ánh xạ lớp: {analysis_result.get('error_detail','')}. Vui lòng kiểm tra cấu hình tên lớp và ánh xạ trong code.")
            if analysis_result.get('confidence') is not None:
                st.info(f"Độ tin cậy (nếu có): {analysis_result['confidence']*100:.2f}%")
        elif analysis_result.get("status") == 'MODEL_ERROR':
            st.error(f"Lỗi Model: {analysis_result.get('error_detail','')}.")
        elif analysis_result.get("status") == 'PREDICTION_ERROR':
             st.error(f"Lỗi phân tích ảnh: {analysis_result.get('error_detail','')}.")
        elif analysis_result.get("status") == 'IMAGE_ERROR':
            st.error(f"Lỗi xử lý ảnh đầu vào: {analysis_result.get('error_detail','')}.")
        else:
            if model is not None:
                 st.info("Chào mừng! Vui lòng chọn ảnh lá cà chua từ máy tính hoặc cung cấp URL để bắt đầu chẩn đoán.")

st.markdown("---")
st.caption("Ứng dụng Chẩn đoán bệnh lá cà chua (Deep Learning - YOLOv8) | Phát triển bởi Nhóm 9")
