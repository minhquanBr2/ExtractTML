from typing import Callable, Tuple
import numpy as np
import cv2
from .config import OCR_WHITELIST


def build_ocr() -> Callable[[np.ndarray], Tuple[str, float]]:
    """Returns a function ocr(img_bin_or_gray)->(text, confidence)."""

    # 1) Try pytesseract
    try:
        import pytesseract  # type: ignore

        def ocr_tesseract(img: np.ndarray) -> Tuple[str, float]:
            if img.ndim == 2:
                proc = img
            else:
                proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            config = f'--oem 3 --psm 8 -c tessedit_char_whitelist={OCR_WHITELIST}'
            data = pytesseract.image_to_data(
                proc, config=config, output_type=pytesseract.Output.DICT
            )

            words = []
            confs = []
            for w, c in zip(data.get("text", []), data.get("conf", [])):
                w = (w or "").strip()
                try:
                    cf = float(c)
                except Exception:
                    cf = -1.0
                if w:
                    words.append(w)
                    if cf >= 0:
                        confs.append(cf)

            text = " ".join(words).strip()
            conf = float(np.mean(confs)) / 100.0 if confs else 0.0
            return text, conf

        print("\n[OCR] Using Tesseract.")
        return ocr_tesseract

    except Exception:
        pass

    # 2) PaddleOCR
    try:
        from paddleocr import PaddleOCR  # type: ignore

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def ocr_paddle(img: np.ndarray) -> Tuple[str, float]:
            if img.ndim == 2:
                proc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                proc = img

            res = ocr.ocr(proc, cls=True)
            if not res or len(res) == 0 or res[0] is None:
                return "", 0.0

            items = res[0]
            if not items:
                return "", 0.0

            texts = []
            confs = []
            for item in items:
                if not item or len(item) < 2:
                    continue
                txt_conf = item[1]
                if not txt_conf or len(txt_conf) < 2:
                    continue
                txt = str(txt_conf[0]).strip()
                cf = float(txt_conf[1]) if txt_conf[1] is not None else 0.0
                if txt:
                    texts.append(txt)
                    confs.append(cf)

            text = " ".join(texts).strip()
            conf = float(np.mean(confs)) if confs else 0.0
            return text, conf

        print("\n[OCR] Using PaddleOCR.")
        return ocr_paddle

    except Exception:
        pass

    # 3) EasyOCR
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=False)

        def ocr_easy(img: np.ndarray) -> Tuple[str, float]:
            if img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = reader.readtext(rgb, detail=1, paragraph=False)
            if not res:
                return "", 0.0
            best = max(res, key=lambda x: float(x[2]))
            text = str(best[1]).strip()
            conf = float(best[2])
            return text, conf

        print("\n[OCR] Using EasyOCR.")
        return ocr_easy

    except Exception:
        pass

    raise RuntimeError(
        "No OCR backend available.\n"
        "Install one of:\n"
        "  - pytesseract + Tesseract engine\n"
        "  - paddleocr + paddlepaddle\n"
        "  - easyocr\n"
    )
