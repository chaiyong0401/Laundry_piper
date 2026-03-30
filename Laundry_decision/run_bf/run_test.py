
import time, numpy as np
import pyrealsense2 as rs
p = rs.pipeline()
cfg = rs.config()
# D435i 표준: 깊이/컬러 640x480@30
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = p.start(cfg)

# 초기 워밍업(센서 안정화)
for _ in range(15):
    p.wait_for_frames(10000)

ok_depth = ok_color = False
t0 = time.time()
while time.time() - t0 < 10:
    fs = p.wait_for_frames(10000)
    d = fs.get_depth_frame()
    c = fs.get_color_frame()
    ok_depth |= bool(d)
    ok_color |= bool(c)
    if ok_depth and ok_color:
        break
p.stop()
print("depth:", ok_depth, "color:", ok_color)

