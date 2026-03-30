#!/usr/bin/env python3
import json, pyrealsense2 as rs

W, H, FPS = 640, 480, 30
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

# 컬러 스트림 프로파일 → intrinsics
cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr  = cprof.get_intrinsics()  # intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height, intr.model, intr.coeffs

out = {
    "fx": intr.fx, "fy": intr.fy,
    "ppx": intr.ppx, "ppy": intr.ppy,
    "width": intr.width, "height": intr.height,
    "model": int(intr.model), "coeffs": list(intr.coeffs)
}
with open("intrinsics_color_640x480.json", "w") as f:
    json.dump(out, f, indent=2)
print(out)

pipe.stop()