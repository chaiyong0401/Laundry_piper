#!/usr/bin/env python3
# run_rgb_segment.py
# 실시간 RealSense - RGB 기반 의류 분리 데모
# author : Seung-yeon Cheon

import cv2, time, glob, argparse
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from sklearn.cluster import KMeans
from collections import defaultdict

# ──────────────────────────[ 설정 ]────────────────────────── #
CFG = {
    "wm": {  # 세탁기
        "ref_pcd": "reference/washer.pcd",  # 드럼 내부 기준 PCD
        "z_crop":  (0.30, 0.99),            # ref PCD 미리 자르기
        "z_range": (0.47, 0.99),            # 실시간 포인트 필터
        "n_clusters": 3                     # 색상 클러스터 개수
    },
    "basket": {  # 빨래바구니
        "ref_pcd": "reference/basket.pcd",
        "z_crop":  (0.00, 1.00),
        "z_range": (0.10, 0.60),
        "n_clusters": 3
    }
}

# ──────────────────────────[ 유틸 함수 ]────────────────────────── #
def load_and_crop_pcd(path, zmin, zmax):
    """reference PCD 불러와 z 범위로 자르기"""
    pcd  = o3d.io.read_point_cloud(path)
    pts  = np.asarray(pcd.points)
    mask = (pts[:,2] >= zmin) & (pts[:,2] <= zmax)
    pcd.points = o3d.utility.Vector3dVector(pts[mask])
    return pcd

def rs_pointcloud_with_color(pipeline, z_range):
    """
    RealSense → (N,3) points, (N,3) rgb 반환
    z_range : (zmin, zmax) 자르기
    """
    align  = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned= align.process(frames)
    depth  = aligned.get_depth_frame()
    color  = aligned.get_color_frame()
    if not depth or not color: return None, None

    pc = rs.pointcloud()
    pc.map_to(color)
    points_rs = pc.calculate(depth)

    vtx = np.asarray(points_rs.get_vertices()).view(np.float32).reshape(-1,3)
    tex = np.asarray(points_rs.get_texture_coordinates()).view(np.float32).reshape(-1,2)
    img = np.asarray(color.get_data())          # BGR
    img = img[..., ::-1]                        # RGB

    h, w, _ = img.shape
    rgb = []
    for u,v in tex:
        x = min(int(u*w), w-1)
        y = min(int(v*h), h-1)
        rgb.append(img[y,x]/255.0)
    rgb = np.asarray(rgb)

    zmin, zmax = z_range
    mask = (vtx[:,2] >= zmin) & (vtx[:,2] <= zmax) & np.isfinite(vtx[:,2])
    return vtx[mask], rgb[mask]

def crop_by_aabb(points, ref_pcd):
    """reference PCD로부터 AABB를 얻어 current points를 crop"""
    aabb = ref_pcd.get_axis_aligned_bounding_box()
    mb   = aabb.get_min_bound(); mb[2] = mb[2]   # 그대로
    aabb = o3d.geometry.AxisAlignedBoundingBox(mb, aabb.get_max_bound())
    curr = o3d.geometry.PointCloud()
    curr.points = o3d.utility.Vector3dVector(points)
    cropped = curr.crop(aabb)
    return np.asarray(cropped.points)

def kmeans_color_seg(points, colors, n_clusters=3, min_pts=200):
    """
    RGB 값 자체를 기준으로 KMeans 군집화하여
    작은 클러스터를 의류로 간주함
    """
    if points.shape[0] == 0:
        return [], []

    # ✅ RGB 값 기준으로 클러스터링 (값은 [0,1] 범위)
    km = KMeans(n_clusters=n_clusters, n_init="auto").fit(colors)
    labels = km.labels_

    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)

    garment_clusters = []
    garment_centers  = []
    for idxs in clusters.values():
        if len(idxs) < min_pts:  # 작은 색상 군집만 선택
            garment_clusters.append(points[idxs])
            garment_centers.append(points[idxs].mean(axis=0))

    return garment_clusters, garment_centers

def visualize(ref_pts, curr_pts, garment_clusters, centers):
    objs = []

    def make(points, c):
        p=o3d.geometry.PointCloud()
        p.points=o3d.utility.Vector3dVector(points)
        p.paint_uniform_color(c); objs.append(p)

    if len(ref_pts):     make(ref_pts, [0,0,1])      # 파랑 기준
    if len(curr_pts):    make(curr_pts, [0.6,0.6,0.6])
    colors_cycle=[[1,0,0],[1,0.5,0],[0,1,1]]
    for i,cl in enumerate(garment_clusters):
        make(cl, colors_cycle[i%len(colors_cycle)])
    for c in centers:
        s=o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s.translate(c); s.paint_uniform_color([0,0,0]); objs.append(s)
    axes=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    objs.append(axes)
    o3d.visualization.draw_geometries(objs)

# ──────────────────────────[ 메인 루틴 ]────────────────────────── #
def run(mode:str):
    cfg = CFG[mode]
    print(f"[INFO] mode={mode}")

    # (1) reference PCD (AABB 용)
    ref_pcd = load_and_crop_pcd(cfg["ref_pcd"], *cfg["z_crop"])
    ref_pts = np.asarray(ref_pcd.points)

    # (2) RealSense 설정
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16,30)
    rs_cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8,30)
    pipeline.start(rs_cfg)

    label_txt, label_col = "Press 'd' to detect", (0,0,0)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            rgbimg = np.asanyarray(color.get_data())
            cv2.putText(rgbimg, label_txt,(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,label_col,3)
            cv2.imshow("RGB Stream", rgbimg)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break

            if key==ord('d'):
                t0=time.time()
                pts, cols = rs_pointcloud_with_color(pipeline, cfg["z_range"])
                if pts is None: continue

                # AABB crop (세탁기 전용)
                if mode=="wm": pts = crop_by_aabb(pts, ref_pcd)

                # 색상 기반 분리
                clusters, centers = kmeans_color_seg(
                    pts, cols, n_clusters=cfg["n_clusters"])

                if centers:
                    print(f"[INFO] 의류 클러스터 {len(centers)}개 탐지")
                    for i,c in enumerate(centers):
                        print(f"  · grasp {i}: {np.round(c,3)}")
                    label_txt, label_col = f"CLOTH {len(centers)}", (0,0,255)
                else:
                    print("[INFO] 의류 미탐지")
                    label_txt, label_col = "EMPTY", (0,255,0)

                t1=time.time()
                print(f"detect_time = {t1-t0:.2f}s")
                visualize(ref_pts, pts, clusters, centers)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ──────────────────────────[ entry ]────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["wm","basket"],required=True)
    args = parser.parse_args()
    run(args.mode)
