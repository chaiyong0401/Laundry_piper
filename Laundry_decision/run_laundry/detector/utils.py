# detector/utils.py
import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import cv2
from sklearn.cluster import KMeans

def assign_rgb_to_points(points, color_image, intr):
    """
    3D 포인트에 대응되는 RGB 값을 할당
    """
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    us = (xs * intr.fx / zs + intr.ppx).astype(int)
    vs = (ys * intr.fy / zs + intr.ppy).astype(int)
    us = np.clip(us, 0, color_image.shape[1] - 1)
    vs = np.clip(vs, 0, color_image.shape[0] - 1)
    colors = color_image[vs, us]  # shape = (N, 3)
    return colors

def get_intrinsics(pipeline):
    """
    RealSense 파이프라인에서 depth 카메라 intrinsics 추출
    """
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        raise RuntimeError("Depth frame을 가져올 수 없음")
    
    intr = rs.video_stream_profile(depth.profile).get_intrinsics()
    return intr

def split_cluster_by_color(cluster_points, cluster_colors, method='kmeans', k=2):
    """
    클러스터 내 RGB 기반 색상 분리 (기본: k-means, k=2)
    """
    if method == 'kmeans':
        km = KMeans(n_clusters=k, random_state=42).fit(cluster_colors)
        labels = km.labels_
    else:
        raise NotImplementedError("지원되지 않는 분할 방식")

    subclusters = []
    for label in np.unique(labels):
        subclusters.append(cluster_points[labels == label])
    return subclusters

# def get_current_pointcloud(pipeline, depth_crop=(0.3, 0.90)):
#     align = rs.align(rs.stream.color)
#     frames = pipeline.wait_for_frames()
#     aligned = align.process(frames)
#     depth = aligned.get_depth_frame()
#     color = aligned.get_color_frame()

#     pc = rs.pointcloud()
#     pc.map_to(color)
#     points = pc.calculate(depth)

#     vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
#     tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
#     img = np.asanyarray(color.get_data())

#     h, w, _ = img.shape
#     colors = []
#     for u, v in tex:
#         x, y = min(int(u * w), w - 1), min(int(v * h), h - 1)
#         colors.append(img[y, x] / 255.0)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(vtx)
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

#     pts = np.asarray(pcd.points)
#     mask = (pts[:, 2] > depth_crop[0]) & (pts[:, 2] < depth_crop[1])
#     pcd.points = o3d.utility.Vector3dVector(pts[mask])
#     if pcd.has_colors():
#         col = np.asarray(pcd.colors)[mask]
#         pcd.colors = o3d.utility.Vector3dVector(col)

#     return np.asarray(pcd.points)
def get_current_pointcloud(pipeline, depth_crop=(0.3, 0.99)):
    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth = aligned.get_depth_frame()
    color = aligned.get_color_frame()

    intr = rs.video_stream_profile(depth.profile).get_intrinsics()

    pc = rs.pointcloud()
    pc.map_to(color)
    points = pc.calculate(depth)

    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    img = np.asanyarray(color.get_data())

    h, w, _ = img.shape
    colors = []
    for u, v in tex:
        x, y = min(int(u * w), w - 1), min(int(v * h), h - 1)
        colors.append(img[y, x] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] > depth_crop[0]) & (pts[:, 2] < depth_crop[1])
    pcd.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        col = np.asarray(pcd.colors)[mask]
        pcd.colors = o3d.utility.Vector3dVector(col)

    return np.asarray(pcd.points), img, intr  # ✅ intrinsics 추가


def load_and_crop_pcd(path, zmin=0.47, zmax=0.99):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    pcd.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return pcd

def align_icp(source_np, target_np):
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_np)
    tgt.points = o3d.utility.Vector3dVector(target_np)
    threshold = 0.05
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T_icp = result.transformation               # <<< 4x4 행렬

    aligned = src.transform(result.transformation)
    return np.asarray(aligned.points), T_icp

def to_open3d_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    numpy (N, 3) 배열을 open3d.geometry.PointCloud로 변환
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("input != numpy.ndarray")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points != (N, 3)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def split_clusters_by_color_lab(clusters, final_pts, color_image, intr, delta_e_thresh=15):
    """
    Lab 색공간 기반 클러스터 분리 함수

    Args:
        clusters: List[np.ndarray], 기존 클러스터 목록 (각 클러스터는 (N_i, 3) shape)
        final_pts: np.ndarray, 전체 포인트 (N, 3)
        color_image: np.ndarray, RGB 이미지 (H, W, 3)
        intr: RealSense intrinsics 객체
        delta_e_thresh: float, Lab 거리 임계값 (기본 15)
    
    Returns:
        same_clusters: ΔE 이내 클러스터 목록
        diff_clusters: ΔE 이상 클러스터 목록
    """

    # 1. 모든 final_pts에 대해 RGB 값 추출
    xs, ys, zs = final_pts[:, 0], final_pts[:, 1], final_pts[:, 2]
    us = (xs * intr.fx / zs + intr.ppx).astype(int)
    vs = (ys * intr.fy / zs + intr.ppy).astype(int)
    us = np.clip(us, 0, color_image.shape[1] - 1)
    vs = np.clip(vs, 0, color_image.shape[0] - 1)
    pt_colors = color_image[vs, us]  # shape (N, 3)
    
    # 2. 클러스터별 평균 RGB 계산
    mean_colors = []
    idx = 0
    for cl in clusters:
        L = len(cl)
        if L == 0:
            mean_colors.append(np.array([0, 0, 0]))
        else:
            mean_colors.append(pt_colors[idx:idx + L].mean(axis=0))
        idx += L

    # 3. Lab 색공간 변환
    mean_colors = np.array(mean_colors, dtype=np.uint8).reshape(-1, 1, 3)
    lab_colors = cv2.cvtColor(mean_colors, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # 4. ΔE 거리 계산 (기준은 첫 번째 클러스터)
    ref = lab_colors[0]
    dists = np.linalg.norm(lab_colors - ref, axis=1)
    print(dists)
    # 5. ΔE 기반 분리
    same_clusters = [clusters[i] for i in range(len(clusters)) if dists[i] <= delta_e_thresh]
    diff_clusters = [clusters[i] for i in range(len(clusters)) if dists[i] > delta_e_thresh]

    return same_clusters, diff_clusters

