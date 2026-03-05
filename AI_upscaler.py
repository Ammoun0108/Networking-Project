import cv2

class AIUpscaler:
    """
    Local FSRCNN upscaler using OpenCV dnn_superres.
    Requires FSRCNN_x2.pb in the same folder (or provide a path).
    """

    def __init__(self, model_path="FSRCNN_x2.pb", target_w=1920, target_h=1080):
        self.target_w = target_w
        self.target_h = target_h

        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError(
                "cv2.dnn_superres not found. Install opencv-contrib-python:\n"
                "  pip install opencv-contrib-python"
            )

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel("fsrcnn", 2)

        print(f"[AI] Local FSRCNN loaded: {model_path}")

    def upscale_to_1080(self, frame_bgr):
        # 480p -> ~960p via FSRCNN x2
        up = self.sr.upsample(frame_bgr)
        # 960p -> 1080p final resize (small stretch)
        up = cv2.resize(up, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)
        return up
