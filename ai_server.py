from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2

TARGET_W, TARGET_H = 1920, 1080

app = FastAPI(title="Upscaler (OpenCV sharpen)")

print("[AI] Simple OpenCV upscaler server starting...")


def upscale_and_sharpen(img_bgr):
    # 1) Upscale to 1080p using bicubic
    up = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

    # 2) Optional light denoise (helps with blocky low-res video)
    up = cv2.bilateralFilter(up, d=7, sigmaColor=30, sigmaSpace=30)

    # 3) Unsharp mask (sharpen edges)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)

    # 4) Watermark so you know this path is used
    cv2.putText(
        sharp,
        "UPSCALED + SHARPENED 1080p",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        3,
    )
    return sharp


@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        # If decode fails, just send back original bytes
        return Response(content=data, media_type="image/jpeg")

    out = upscale_and_sharpen(img)

    ok, enc = cv2.imencode(".jpg", out)
    if not ok:
        return Response(content=data, media_type="image/jpeg")

    return Response(content=enc.tobytes(), media_type="image/jpeg")


