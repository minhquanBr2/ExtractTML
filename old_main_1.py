# main.py
import os
import argparse
import json
from google import genai
from PIL import Image
# Định nghĩa Bộ Quy Tắc Có Sẵn
# KEY: Mô tả để người dùng chọn. VALUE: Mô tả chi tiết để gửi cho Gemini.
RULE_OPTIONS = {
    1: "VÒNG TRÒN có VIỀN ĐỎ để trích xuất Mã TML.",
    2: "VÒNG TRÒN có VIỀN TÍM để trích xuất Mã TML.",
    3: "VÒNG TRÒN có VIỀN XANH để trích xuất Mã TML.",
    4: "VÒNG TRÒN có VIỀN CAM để trích xuất Mã TML.",
    5: "VÒNG TRÒN có VIỀN ĐEN để trích xuất Mã TML.",   
    6: "HÌNH VUÔNG/HÌNH CHỮ NHẬT có VIỀN ĐỎ để trích xuất nội dung.",
    7: "HÌNH VUÔNG/HÌNH CHỮ NHẬT có VIỀN ĐEN để trích xuất nội dung.",
    8: "HÌNH VUÔNG/HÌNH CHỮ NHẬT có VIỀN XANH để trích xuất nội dung.",
    9: "HÌNH VUÔNG/HÌNH CHỮ NHẬT có NỀN VÀNG để trích xuất nội dung.",  
    10: "HÌNH TỨ GIÁC có VIỀN XANH để trích xuất nội dung.",
    11: "HÌNH NGŨ GIÁC có VIỀN XANH để trích xuất nội dung.",
    12: "HÌNH LỤC GIÁC có VIỀN XANH để trích xuất nội dung."
}

# 1. Khởi tạo Client Gemini và Cấu hình
# Model cho Vision (đa phương thức)
MODEL_NAME = "gemini-2.5-pro" 

# --- THIẾT LẬP VÀ KHỞI TẠO CLIENT ---
def initialize_client():
    """Khởi tạo API và kiểm tra API Key."""
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCJnvsu-XwFRsPmE30ldk_UkUs0ojSyOz4"
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("LỖI: Biến môi trường 'GOOGLE_API_KEY' chưa được thiết lập.")
    
    try:
        client = genai.Client()
        print("✅ Đã khởi tạo mô hình thành công.")
        return client
    except Exception as e:
        raise ConnectionError(f"LỖI KẾT NỐI: Không thể khởi tạo Client: {e}")

# Thay thế hàm load_examples trong main.py

def load_examples(file_path: str) -> list:
    """Đọc file JSON ví dụ, tải các ảnh ví dụ và định dạng nội dung Few-shot."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            examples = json.load(f)
        
        # Danh sách nội dung (content list) sẽ được truyền trực tiếp vào API call
        content_list = []
        
        for i, ex in enumerate(examples):
            # 1. Tải ảnh ví dụ
            try:
                example_img = Image.open(ex['example_image_path'])
            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy ảnh ví dụ: {ex['example_image_path']}")
                continue

            # 2. Thêm Text Instruction (hướng dẫn)
            content_list.append(f"--- VÍ DỤ {i+1} ---")
            content_list.append(f"RULE: Đây là ví dụ về đối tượng {ex['description']}")
            
            # 3. Thêm Ảnh ví dụ
            content_list.append(example_img)
            
            # 4. Thêm Expected Output (Kết quả mong muốn)
            content_list.append("KẾT QUẢ MONG MUỐN CHO HÌNH ẢNH TRÊN:")
            # Đảm bảo kết quả mong muốn là chuỗi JSON sạch
            content_list.append(json.dumps(ex['expected_output_json'], indent=2, ensure_ascii=False))

        return content_list
    
    except Exception as e:
        print(f"Lỗi khi tải hoặc định dạng ví dụ huấn luyện: {e}")
        return []

def build_dynamic_prompt(selected_rules: list) -> str:
    """Xây dựng Prompt chi tiết dựa trên lựa chọn của người dùng, áp dụng ràng buộc JSON nghiêm ngặt."""
    if not selected_rules:
        return ""

    rule_descriptions = "\n".join([f"- {RULE_OPTIONS[i]}" for i in selected_rules])
    
    return f"""
Bạn là một công cụ phân tích đồ họa chuyên nghiệp. Nhiệm vụ của bạn là kiểm tra hình ảnh đính kèm (là trang [X] của tài liệu) và trích xuất tất cả các hình học sau: hình tròn, hình vuông, hình chữ nhật, hình lục giác, hình ngũ giác, hình đa giác khác.
Đối với mỗi hình dạng được phát hiện, hãy xác định màu sắc chính của nó (ví dụ: đỏ, xanh lá cây, vàng, đen
"""

def extract_data_with_selection(client: genai.Client,image_path: str, selected_rules_str: str):
    """Thực hiện trích xuất dựa trên lựa chọn quy tắc của người dùng."""
    
    try:
        selected_rules_indices = [int(i.strip()) for i in selected_rules_str.split(',') if i.strip().isdigit()]
    except Exception:
        print("Lỗi: Định dạng lựa chọn không hợp lệ. Vui lòng nhập các số cách nhau bằng dấu phẩy.")
        return

    if not selected_rules_indices:
        print("Lỗi: Vui lòng chọn ít nhất một quy tắc hợp lệ (1-8).")
        return

    # Lọc ra các quy tắc hợp lệ và xây dựng Prompt
    valid_rules = [i for i in selected_rules_indices if i in RULE_OPTIONS]
    prompt_text = build_dynamic_prompt(valid_rules)
    
    if not prompt_text:
        print("Không có quy tắc hợp lệ nào được chọn.")
        return

    # Tải ảnh
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return

    print("--- Khởi chạy Trích xuất Dữ liệu Động ---")
    print(f"1. Đang tải ảnh: {os.path.basename(image_path)}")
    print(f"2. Áp dụng {len(valid_rules)} quy tắc.")

    #print(f"2. Áp dụng {len(valid_rules)} quy tắc. Huấn luyện Hình ảnh: {'Có' if examples_content else 'Không'}.")
    #print(f"3. Gọi API Gemini với model: {MODEL_NAME}")
    
    # Gọi API Gemini
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_text, img]
            #contents=contents # Truyền danh sách nội dung hỗn hợp
        )
    # Tách khối code JSON ra khỏi phản hồi thô
        raw_text = response.text.strip()
        if raw_text.startswith('```json') and raw_text.endswith('```'):
            json_string = raw_text[len('```json'):-len('```')].strip()
        else:
            json_string = raw_text # Thử tải ngay cả khi không có code block
        
        # Hiển thị kết quả
        print("\n================ KẾT QUẢ TRÍCH XUẤT ĐỘNG ================")
        try:
            json_output = json.loads(json_string)
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("Lỗi định dạng JSON. Phản hồi thô của AI:")
            print(raw_text)
        print("=====================================================")

    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình gọi API: {e}")

if __name__ == "__main__":
    gemini_client = initialize_client()
    parser = argparse.ArgumentParser(description="Công cụ Trích xuất Dữ liệu Bản vẽ Tùy chỉnh sử.")
    parser.add_argument("--image", required=True, help="Đường dẫn tới file ảnh bản vẽ kỹ thuật.")  
    parser.add_argument("--rules", required=True, 
                        help="Chọn các quy tắc cần áp dụng (cách nhau bằng dấu phẩy), ví dụ: 1,3,7")
    #parser.add_argument("--examples", required=False, default=None,
                        #help="[Tùy chọn] Đường dẫn tới file JSON chứa các ví dụ huấn luyện (Visual Few-shot learning).")
                        
    # Hiển thị các lựa chọn cho người dùng
    print("\n--- CÁC QUY TẮC TRÍCH XUẤT CÓ SẴN ---")
    for key, value in RULE_OPTIONS.items():
        print(f"{key}. {value}")
    print("--------------------------------------\n")

    args = parser.parse_args()
    
    #extract_data_with_selection(gemini_client,args.image,args.rules, args.examples)
    extract_data_with_selection(gemini_client,args.image,args.rules)
