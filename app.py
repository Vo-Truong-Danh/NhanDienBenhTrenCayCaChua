import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Dự đoán bệnh lá cà chua", layout="wide")
st.title("🧪 Dự đoán bệnh lá Cà Chua qua ảnh")
st.write(
    "Tải lên ảnh lá cà chua, hệ thống sẽ tự động phân tích bệnh và đưa ra gợi ý. "
    "Lưu ý: Model chuyên dùng cho lá cà chua và có thể cho kết quả không chính xác với các loại ảnh khác."
)


# THÔNG TIN BỆNH VÀ CÁCH XỬ LÝ 

disease_info = {
    'Bacterial_spot': {
        'scientific_name_en': "Xanthomonas spp. (e.g., X. campestris pv. vesicatoria)",
        'vietnamese_name': "Bệnh đốm khuẩn (do Xanthomonas spp.)",
        'remedies': [
            "Sử dụng thuốc trừ bệnh gốc Đồng (ví dụ: Copper Oxychloride, Copper Hydroxide) theo nồng độ khuyến cáo.",
            "Trong trường hợp áp lực bệnh cao, có thể xem xét sử dụng kháng sinh chuyên dùng (ví dụ: Streptomycin, Kasugamycin) nhưng phải tuân thủ nghiêm ngặt liều lượng, thời gian cách ly và chỉ sử dụng khi thực sự cần thiết để tránh kháng thuốc."
        ],
        'actions': [
            "Vệ sinh đồng ruộng: Thu gom và tiêu hủy ngay các bộ phận cây bị bệnh (lá, cành, quả).",
            "Luân canh cây trồng: Không trồng cà chua hoặc các cây cùng họ (ớt, khoai tây) trên cùng một chân đất trong ít nhất 2-3 năm.",
            "Giống kháng bệnh: Ưu tiên chọn giống cà chua có khả năng kháng bệnh đốm khuẩn.",
            "Quản lý tưới tiêu: Tránh tưới nước trực tiếp lên lá, đặc biệt vào buổi chiều tối. Giữ cho bề mặt lá khô ráo.",
            "Khử trùng dụng cụ: Thường xuyên vệ sinh và khử trùng các dụng cụ làm vườn (dao, kéo, cuốc).",
            "Mật độ trồng hợp lý: Trồng với mật độ vừa phải để đảm bảo độ thông thoáng cho vườn cây."
        ]
    },
    'Early_blight': {
        'scientific_name_en': "Alternaria solani",
        'vietnamese_name': "Bệnh cháy sớm (do Alternaria solani)",
        'remedies': [
            "Sử dụng thuốc trừ nấm có hoạt chất Mancozeb, Chlorothalonil khi bệnh mới xuất hiện.",
            "Các hoạt chất Azoxystrobin, Difenoconazole cũng cho hiệu quả tốt. Nên luân phiên thuốc để tránh kháng thuốc.",
            "Phun thuốc kỹ cả hai mặt lá và phun nhắc lại theo hướng dẫn của nhà sản xuất."
        ],
        'actions': [
            "Vệ sinh đồng ruộng: Dọn sạch tàn dư cây trồng vụ trước, đặc biệt là những cây bị bệnh.",
            "Thoát nước tốt: Đảm bảo ruộng cà chua không bị úng nước, nhất là trong mùa mưa.",
            "Luân canh cây trồng: Thực hiện luân canh với các cây trồng khác họ.",
            "Bón phân cân đối: Tránh bón thừa đạm (N), tăng cường bón kali (K) và canxi (Ca) để cây cứng cáp.",
            "Cắt tỉa lá bệnh: Tỉa bỏ các lá già, lá bị bệnh ở gốc để giảm nguồn bệnh và tạo độ thông thoáng."
        ]
    },
    'Late_blight': {
        'scientific_name_en': "Phytophthora infestans",
        'vietnamese_name': "Bệnh sương mai (do Phytophthora infestans)",
        'remedies': [
            "Phun thuốc phòng trừ khi thời tiết thuận lợi cho bệnh phát triển (ẩm độ cao, mưa nhiều, có sương mù), đặc biệt ở giai đoạn cây ra hoa, đậu quả.",
            "Sử dụng các loại thuốc có hoạt chất như: Mancozeb + Metalaxyl, Cymoxanil + Mancozeb, Propamocarb, Dimethomorph.",
            "Luân phiên các nhóm thuốc khác nhau để hạn chế sự hình thành tính kháng của nấm bệnh."
        ],
        'actions': [
            "Chọn giống kháng: Sử dụng giống cà chua có khả năng kháng bệnh sương mai.",
            "Mật độ trồng: Trồng thưa, hợp lý để vườn luôn thông thoáng.",
            "Quản lý nước: Thoát nước tốt cho ruộng, tránh để nước đọng lại sau mưa hoặc tưới.",
            "Vệ sinh vườn: Tiêu hủy kịp thời những cây, lá bị bệnh nặng.",
            "Luân canh: Không trồng cà chua liên tục nhiều năm trên một thửa ruộng."
        ]
    },
    'Leaf_Mold': {
        'scientific_name_en': "Fulvia fulva (syn. Cladosporium fulvum)",
        'vietnamese_name': "Bệnh mốc lá (do Fulvia fulva)",
        'remedies': [
            "Sử dụng thuốc trừ nấm gốc Đồng, hoặc các hoạt chất như Chlorothalonil, Azoxystrobin, Trifloxystrobin.",
            "Phun thuốc kỹ vào mặt dưới của lá, nơi nấm bệnh thường phát triển mạnh."
        ],
        'actions': [
            "Thông gió: Đảm bảo độ thông thoáng tốt, đặc biệt quan trọng trong điều kiện nhà kính hoặc nhà lưới.",
            "Kiểm soát độ ẩm: Giảm độ ẩm không khí bằng cách tưới nước hợp lý, tránh tưới vào buổi chiều tối.",
            "Cắt tỉa: Loại bỏ lá già, lá gốc và những lá bị bệnh để giảm nguồn lây nhiễm.",
            "Vệ sinh: Dọn dẹp tàn dư thực vật bị bệnh."
        ]
    },
    'Septoria_leaf_spot': {
        'scientific_name_en': "Septoria lycopersici",
        'vietnamese_name': "Bệnh đốm lá Septoria (do Septoria lycopersici)",
        'remedies': [
            "Phun thuốc trừ nấm chứa hoạt chất Chlorothalonil, Mancozeb khi triệu chứng bệnh xuất hiện.",
            "Thuốc gốc Đồng cũng có tác dụng phòng trừ nhất định."
        ],
        'actions': [
            "Tiêu hủy lá bệnh: Thu gom và tiêu hủy các lá bị nhiễm bệnh để giảm thiểu sự lây lan.",
            "Vệ sinh đồng ruộng: Giữ cho vườn cà chua sạch sẽ, không có tàn dư cây bệnh.",
            "Luân canh: Áp dụng chế độ luân canh cây trồng ít nhất 1-2 năm với cây không phải là ký chủ của nấm Septoria.",
            "Phương pháp tưới: Ưu tiên tưới gốc, tránh tưới phun lên lá làm ẩm lá kéo dài.",
            "Dinh dưỡng: Bón phân cân đối, tăng cường phân hữu cơ và kali để cây khỏe mạnh, tăng sức đề kháng."
        ]
    },
    'Spider_mites Two-spotted_spider_mite': {
        'scientific_name_en': "Tetranychus urticae",
        'vietnamese_name': "Nhện đỏ hai chấm (Tetranychus urticae)",
        'remedies': [
            "Sử dụng thuốc đặc trị nhện như Abamectin, Emamectin Benzoate, Spiromesifen, Hexythiazox.",
            "Có thể dùng các sản phẩm sinh học như dầu khoáng, nấm ký sinh (Beauveria bassiana, Metarhizium anisopliae).",
            "Phun kỹ mặt dưới lá, nơi nhện thường tập trung. Phun lặp lại sau 5-7 ngày nếu mật độ nhện cao."
        ],
        'actions': [
            "Biện pháp cơ học: Phun nước mạnh vào mặt dưới lá (khi mật độ nhện còn thấp) để rửa trôi nhện.",
            "Bảo tồn thiên địch: Tạo điều kiện cho các loài thiên địch của nhện phát triển (ví dụ: bọ rùa, bọ cánh gân, nhện bắt mồi).",
            "Vệ sinh vườn: Cắt tỉa và tiêu hủy các lá, cành bị nhện hại nặng.",
            "Tránh khô hạn: Duy trì độ ẩm thích hợp cho vườn, vì nhện đỏ thường phát triển mạnh trong điều kiện khô nóng."
        ]
    },
    'Target_Spot': {
        'scientific_name_en': "Corynespora cassiicola",
        'vietnamese_name': "Bệnh đốm mắt cua (do Corynespora cassiicola)",
        'remedies': [
            "Sử dụng thuốc trừ nấm có hoạt chất Chlorothalonil, Mancozeb.",
            "Các thuốc nhóm Strobilurin (ví dụ: Azoxystrobin, Pyraclostrobin) cũng cho thấy hiệu quả tốt."
        ],
        'actions': [
            "Vệ sinh: Thu dọn và tiêu hủy tàn dư cây bệnh từ vụ trước.",
            "Thông thoáng: Đảm bảo vườn trồng thông thoáng, tránh ẩm độ cao kéo dài.",
            "Luân canh: Thực hiện luân canh với các cây trồng không phải là ký chủ của nấm.",
            "Dinh dưỡng: Bón phân cân đối, không bón thừa đạm."
        ]
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'scientific_name_en': "Tomato yellow leaf curl virus (TYLCV)",
        'vietnamese_name': "Virus xoăn vàng lá cà chua (TYLCV)",
        'remedies': [
            "**Không có thuốc đặc trị bệnh virus.** Biện pháp chủ yếu là phòng trừ côn trùng môi giới.",
            "Kiểm soát bọ phấn trắng (môi giới truyền bệnh): Sử dụng các hoạt chất như Imidacloprid, Thiamethoxam, Dinotefuran, Pymetrozin. Luân phiên thuốc để tránh kháng.",
            "Sử dụng các biện pháp sinh học: Dầu khoáng, xà phòng côn trùng để giảm mật độ bọ phấn."
        ],
        'actions': [
            "Nhổ bỏ và tiêu hủy: Phát hiện sớm và tiêu hủy ngay những cây có triệu chứng bệnh để ngăn chặn lây lan.",
            "Giống kháng Virus: Ưu tiên sử dụng các giống cà chua có khả năng kháng hoặc chống chịu virus TYLCV.",
            "Vệ sinh vườn: Diệt trừ cỏ dại xung quanh vườn, vì cỏ dại có thể là nơi trú ngụ của bọ phấn trắng.",
            "Luân canh: Thực hiện luân canh với cây trồng không phải là ký chủ của virus và bọ phấn.",
            "Nhà lưới/màng chắn: Sử dụng nhà lưới có mắt lưới nhỏ để ngăn chặn bọ phấn xâm nhập, đặc biệt trong giai đoạn cây con."
        ]
    },
    'Tomato_mosaic_virus': {
        'scientific_name_en': "Tomato mosaic virus (ToMV)",
        'vietnamese_name': "Virus khảm lá cà chua (ToMV)",
        'remedies': [
            "**Không có thuốc đặc trị bệnh virus.** Tập trung vào các biện pháp phòng ngừa lây nhiễm.",
            "Kiểm soát côn trùng môi giới (nếu có): Một số virus khảm có thể lây qua côn trùng, cần xác định và kiểm soát (ví dụ: rầy, rệp)."
        ],
        'actions': [
            "Nhổ bỏ và tiêu hủy: Loại bỏ và tiêu hủy ngay cây bị bệnh.",
            "Sử dụng giống kháng: Chọn giống có khả năng kháng virus ToMV.",
            "Vệ sinh dụng cụ: Khử trùng dụng cụ (dao, kéo) thường xuyên bằng cồn y tế (>70%) hoặc dung dịch Javel khi làm việc giữa các cây, các luống.",
            "Hạn chế tiếp xúc cơ học: Virus dễ lây qua vết thương cơ giới, hạn chế các hoạt động gây xây xát cho cây.",
            "Không hút thuốc lá: Người làm vườn không nên hút thuốc lá khi đang làm việc với cây cà chua, vì virus ToMV có thể tồn tại trong thuốc lá và lây nhiễm sang cây."
        ]
    },
    'healthy': {
        'scientific_name_en': "N/A (Healthy Plant)",
        'vietnamese_name': "Cây khỏe mạnh",
        'remedies': ["Không cần xử lý thuốc bệnh. Tiếp tục duy trì các biện pháp chăm sóc tốt."],
        'actions': [
            "Chăm sóc định kỳ: Tưới nước đủ ẩm theo nhu cầu của cây, tránh để cây bị úng hoặc hạn hán.",
            "Bón phân cân đối: Cung cấp đầy đủ và cân đối các chất dinh dưỡng đa, trung, vi lượng.",
            "Thăm vườn thường xuyên: Quan sát cây hàng ngày để phát hiện sớm bất kỳ dấu hiệu bất thường nào của sâu bệnh.",
            "Phòng ngừa tổng hợp: Tiếp tục áp dụng các biện pháp phòng ngừa chung như vệ sinh đồng ruộng, luân canh cây trồng (nếu có kế hoạch cho vụ sau), chọn giống tốt."
        ]
    }
}

# THAM SỐ CẤU HÌNH CHO MODEL
MODEL_PATH = "tomato_cnn_model.h5"
CONFIDENCE_THRESHOLD = 0.6
GREEN_RATIO_THRESHOLD = 0.30 #Điều chỉnh tỷ lệ nhận diện ảnh ko phải cà chua

# LOAD MODEL & LABELS GỐC
@st.cache_resource(show_spinner="Đang tải model nhận dạng...")
def load_model_from_path(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.error(f"Hãy đảm bảo file model '{model_path}' tồn tại trong cùng thư mục với app.py hoặc cung cấp đường dẫn chính xác.")
        return None

@st.cache_data
def load_original_class_names():
    return [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

model = load_model_from_path(MODEL_PATH)
original_class_names = load_original_class_names()
class_name_keys = [name.replace("Tomato___", "") for name in original_class_names]

# HÀM XỬ LÝ
def preprocess_image(image: Image.Image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def check_green_ratio(image: Image.Image):
    img_array = np.array(image.convert('RGB'))
    green_pixels = np.sum((img_array[:,:,1] > img_array[:,:,0]) & \
                          (img_array[:,:,1] > img_array[:,:,2]) & \
                          (img_array[:,:,1] > 30))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    ratio = green_pixels / total_pixels if total_pixels > 0 else 0
    return ratio

def predict_and_analyze(image_input: Image.Image):
    if model is None:
        return 'MODEL_ERROR', None, 0.0

    green_pixel_ratio = check_green_ratio(image_input)
    if green_pixel_ratio < GREEN_RATIO_THRESHOLD:
        return 'LOW_GREEN', f"{green_pixel_ratio*100:.1f}%", 0.0

    try:
        processed_image = preprocess_image(image_input)
        prediction_probabilities = model.predict(processed_image, verbose=0)[0]
        predicted_index = np.argmax(prediction_probabilities)
        confidence = float(prediction_probabilities[predicted_index])
        predicted_class_key = class_name_keys[predicted_index]
        return 'OK', predicted_class_key, confidence
    except Exception as e:
        return 'PREDICTION_ERROR', "Lỗi xử lý model .", 0.0

# GIAO DIỆN 
if model is None:
    st.error("Không thể tải Model. Ứng dụng không thể hoạt động. Vui lòng kiểm tra lại đường dẫn hoặc file model.")
else:
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = {"status": None, "class_key": None, "confidence": None, "error_detail": None, "file_id": None}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🖼️ Ảnh Lá Cà Chua")
        uploaded_file = st.file_uploader(
            "Tải ảnh lên tại đây (tự động phân tích):",
            type=["jpg", "jpeg", "png"],
            key="file_uploader_key" 
        )

        if uploaded_file is not None:
            current_file_id = uploaded_file.file_id 
            if st.session_state.last_analysis["file_id"] != current_file_id:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
                    with st.spinner("Đang phân tích ảnh..."):
                        status, result_data, confidence_or_ratio = predict_and_analyze(image)

                    # Update session state with new analysis results
                    if status == 'LOW_GREEN':
                        st.session_state.last_analysis = {"status": status, "class_key": None, "confidence": None, "error_detail": result_data, "file_id": current_file_id}
                    elif status == 'OK':
                        st.session_state.last_analysis = {"status": status, "class_key": result_data, "confidence": confidence_or_ratio, "error_detail": None, "file_id": current_file_id}
                    else: # MODEL_ERROR or PREDICTION_ERROR
                        st.session_state.last_analysis = {"status": status, "class_key": None, "confidence": None, "error_detail": result_data if result_data else "Lỗi hệ thống không xác định.", "file_id": current_file_id}
                        if status == 'MODEL_ERROR': # Specific message for model error during predict
                             st.error("Lỗi: Model nhận dạng không thể thực hiện dự đoán lúc này.")

                except Exception as e:
                    st.error(f"Lỗi khi mở hoặc xử lý ảnh: {e}")
                    st.session_state.last_analysis = {"status": "IMAGE_ERROR", "class_key": None, "confidence": None, "error_detail": str(e), "file_id": current_file_id}
            else:
                try:
                    image = Image.open(uploaded_file) 
                    st.image(image, caption="Ảnh đã tải lên (kết quả cũ)", use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi hiển thị lại ảnh: {e}")
                    st.session_state.last_analysis = {"status": "IMAGE_ERROR", "class_key": None, "confidence": None, "error_detail": str(e), "file_id": current_file_id}


        elif st.session_state.last_analysis["file_id"] is not None: # File was removed
            st.session_state.last_analysis = {"status": None, "class_key": None, "confidence": None, "error_detail": None, "file_id": None}


    with col2:
        st.header("📊 Kết quả Phân Tích")
        analysis_result = st.session_state.last_analysis

        if analysis_result["status"] == 'OK':
            predicted_class_key = analysis_result["class_key"]
            confidence = analysis_result["confidence"]
            display_confidence = confidence * 100

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"**Độ tin cậy thấp ({display_confidence:.2f}%). Model không chắc chắn về kết quả này.**"
                )
                st.info(
                    "Nguyên nhân có thể là:\n"
                    "- Ảnh chụp chưa rõ nét, thiếu sáng, hoặc góc chụp chưa tối ưu.\n"
                    "- Triệu chứng bệnh không điển hình hoặc bệnh không nằm trong danh mục model được huấn luyện.\n\n"
                    "**Gợi ý:** Vui lòng thử lại với ảnh khác, chụp rõ hơn, hoặc tham khảo ý kiến chuyên gia."
                )
            else:
                info = disease_info.get(predicted_class_key)
                if info:
                    st.success(f"**Bệnh dự đoán: {info['vietnamese_name']}**")
                    st.markdown(f"*(Tên khoa học: {info['scientific_name_en']})*")
                    st.info(f"Độ tin cậy: {display_confidence:.2f}%")

                    if predicted_class_key != 'healthy':
                        st.subheader("⚠️ Gợi ý xử lý và khắc phục:")
                        with st.expander("**Biện pháp hóa học (Tham khảo)**", expanded=True):
                            if info['remedies']:
                                for remedy in info['remedies']:
                                    st.write(f"• {remedy}")
                            else:
                                st.write("• Không có gợi ý thuốc cụ thể cho trường hợp này.")

                        with st.expander("**Biện pháp canh tác và phòng ngừa**", expanded=True):
                            if info['actions']:
                                 for action in info['actions']:
                                    st.write(f"• {action}")
                            else:
                                st.write("• Không có gợi ý biện pháp cụ thể.")
                        st.warning(
                            """
                            **Lưu ý quan trọng (Thuốc BVTV & Biện pháp canh tác):**
                            \nCác thông tin gợi ý chỉ mang tính chất **tham khảo**. Hiệu quả thực tế phụ thuộc vào nhiều yếu tố.
                            \nĐể có giải pháp phù hợp và hiệu quả nhất:
                            \n1. Luôn đọc kỹ và tuân thủ hướng dẫn sử dụng trên nhãn thuốc BVTV.
                            \n2. Áp dụng nguyên tắc 4 đúng và đảm bảo thời gian cách ly.
                            \n3. **Hãy ưu tiên tham vấn ý kiến từ cán bộ kỹ thuật nông nghiệp hoặc chuyên gia bảo vệ thực vật tại địa phương của bạn.**
                            """
                        )
                    else:
                        st.balloons()
                        st.write("🎉 Chúc mừng! Cây cà chua của bạn trông khỏe mạnh.")
                        info_healthy = disease_info.get('healthy')
                        if info_healthy:
                            with st.expander("**Lời khuyên duy trì sức khỏe cho cây**", expanded=True):
                                if info_healthy['actions']:
                                    for action in info_healthy['actions']:
                                        st.write(f"• {action}")
                    st.markdown("---")
                else:
                    st.error(f"Lỗi hệ thống: Không tìm thấy thông tin chi tiết cho mã bệnh '{predicted_class_key}'.")
                    st.info(f"Độ tin cậy (nếu có): {display_confidence:.2f}%")

        elif analysis_result["status"] == 'LOW_GREEN':
            st.error(
                f"Ảnh không phù hợp! Có thể đang không phải đây là ảnh và cà chua ! "
                "Ảnh cần rõ nét hơn và tập trung vào lá cây. Vui lòng thử lại với ảnh khác hoặc liên hệ tác giả để nếu đây là nhầm lẫn"
            )
            st.info("Yêu cầu ảnh chụp rõ lá cà chua, chiếm phần lớn diện tích ảnh, với đủ ánh sáng và nền không quá phức tạp.")
        elif analysis_result["status"] == 'MODEL_ERROR':
            st.error("Lỗi: Model nhận dạng không thể thực hiện dự đoán. Vui lòng kiểm tra thông báo lỗi khi tải model (nếu có).")
        elif analysis_result["status"] == 'PREDICTION_ERROR':
             st.error(f"Lỗi trong quá trình phân tích ảnh. Chi tiết: {analysis_result['error_detail']}. Vui lòng thử lại.")
        elif analysis_result["status"] == 'IMAGE_ERROR':
            st.error(f"Lỗi xử lý ảnh: {analysis_result['error_detail']}. Vui lòng chọn file ảnh hợp lệ (JPG, JPEG, PNG).")
        else:
            if model is not None:
                st.info("Chào mừng bạn! Hãy tải ảnh lá cà chua lên ở cột bên trái để bắt đầu phân tích.")


st.markdown("---")
st.caption("Ứng dụng phân loại bệnh lá cà chua ( Depp-Learning - Nhóm 9 )")