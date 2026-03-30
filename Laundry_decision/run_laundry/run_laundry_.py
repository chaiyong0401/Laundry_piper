'''
python run_laundry_.py \
  --mode file \
  --pcd /home/cheon/Documents/train_all/dataset/dataset_wm/cloth/1/pc/27.pcd \
  --png /home/cheon/Documents/train_all/dataset/dataset_wm/cloth/1/rgb/27_rgb.png \
  --intr_json /home/cheon/Documents/run_bf/intrinsics_color_640x480.json \
  --json --vis
'''
import argparse, time, cv2, numpy as np, pyrealsense2 as rs, open3d as o3d
from PIL import Image
import torch, torchvision.transforms as T
import sys, json, time
from types import SimpleNamespace

# --- 너의 프로젝트 모듈 ---
from detector.rgb_classifier   import RGBOnlyClassifier
from detector.utils            import (get_current_pointcloud, load_and_crop_pcd,
                                       align_icp, to_open3d_pcd, split_clusters_by_color_lab)
from detector.zscore_detector  import (detect_anomalies_adaptive,
                                       cluster_and_get_targets,
                                       visualize_pointcloud_with_garments_open3d, split_colorful_clusters)

# ================== 상수/설정 ==================
Z_MIN, Z_MAX = 0.35, 0.94
CYL_RADIUS = 0.29
FILTER_RADIUS = 0.30

CFG = {
    "wm": {
        "ref_pcd"      : "reference/washer.pcd",
        "z_crop"       : (0.6, 1.2),
        "z_range"      : (Z_MIN, Z_MAX),
        "need_clf"     : True,
        "eps"          : 0.037, "min_samples": 30
    },
    # "basket": {...},
    # "failure": {...}
}

# ================== 유틸 ==================
def load_rgb_model(weight="models/rgb_only_best.pth"):
    model = RGBOnlyClassifier().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()
    tf = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return model, tf

def predict_rgb(model, tf, pipeline):
    frames = pipeline.wait_for_frames()
    color  = frames.get_color_frame()
    if not color: return None, None
    img  = np.asanyarray(color.get_data())
    x    = tf(Image.fromarray(img[..., ::-1])).unsqueeze(0).to(
           "cuda" if torch.cuda.is_available() else "cpu")
    pred = model(x).argmax(1).item()
    return pred, img

def estimate_center_from_pcd(pcd, z_thresh=0.4):
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] > z_thresh) & (pts[:, 2] < 1.0)
    pts_filtered = pts[mask]
    pcd_z = o3d.geometry.PointCloud()
    pcd_z.points = o3d.utility.Vector3dVector(pts_filtered)
    pcd_z, _ = pcd_z.remove_radius_outlier(nb_points=10, radius=0.02)
    pcd_z, _ = pcd_z.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    center = np.mean(np.asarray(pcd_z.points), axis=0)
    return center

def create_cylinder_shell(center_xy, radius, z_min, z_max, radial_res=300, height_res=100, color=[1, 0, 0]):
    cx, cy = center_xy
    zs = np.linspace(z_min, z_max, height_res)
    thetas = np.linspace(0, 2 * np.pi, radial_res)
    points = [[cx + radius * np.cos(t), cy + radius * np.sin(t), z] for z in zs for t in thetas]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.paint_uniform_color(color)
    return pc

def create_z_axis_cylinder(radius, z_min, z_max, radial_res=200, height_res=150, color=[1, 0, 0]):
    zs = np.linspace(z_min, z_max, height_res)
    thetas = np.linspace(0, 2 * np.pi, radial_res)
    points = [[radius * np.cos(t), radius * np.sin(t), z] for z in zs for t in thetas]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.paint_uniform_color(color)
    return pc

def filter_points_inside_radius(points, center_xy, radius):
    x, y = points[:, 0], points[:, 1]
    cx, cy = center_xy
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    return dist <= radius

def run_icp_y_negative_only(curr_pcd, ref_pcd):
    """
    curr_pcd와 ref_pcd의 y<0 부분만 추출해 ICP 정합을 수행
    """
    ref_pts = np.asarray(ref_pcd.points)
    curr_pts = np.asarray(curr_pcd.points)
    ref_mask = ref_pts[:, 1] < 0.0
    curr_mask = curr_pts[:, 1] < 0.0
    ref_half = o3d.geometry.PointCloud()
    curr_half = o3d.geometry.PointCloud()
    ref_half.points = o3d.utility.Vector3dVector(ref_pts[ref_mask])
    curr_half.points = o3d.utility.Vector3dVector(curr_pts[curr_mask])
    result = o3d.pipelines.registration.registration_icp(
        curr_half.voxel_down_sample(0.005),
        ref_half.voxel_down_sample(0.005),
        0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    curr_pcd.transform(result.transformation)
    return curr_pcd, result.transformation

def rotate_cluster(cluster, center, angle_deg):
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    translated = cluster[:, :2] - center[:2]
    rotated = translated @ rot_matrix.T + center[:2]
    return np.hstack([rotated, cluster[:, 2:]])

def camera_to_ee(p_cam: np.ndarray):
    R = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    T = np.array([0.05, 0, -0.05])
    return R @ p_cam + T

def load_intrinsics_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    fx = float(d["fx"]); fy = float(d["fy"])
    ppx = float(d["ppx"]); ppy = float(d["ppy"])
    W = int(d.get("width", 640)); H = int(d.get("height", 480))
    # split_colorful_clusters 가 intr.fx 같은 속성 접근을 할 수 있게 SimpleNamespace로 래핑
    return SimpleNamespace(fx=fx, fy=fy, ppx=ppx, ppy=ppy, width=W, height=H)

# ================== 메인 루틴 ==================
def run(mode: str, json_output: bool,
        pcd_path: str=None, png_path: str=None, intr_json: str=None,
        vis: bool=False):
    if mode == "file":
        # ---------- 파일 1장만 처리 ----------
        print("[INFO] mode = file (single PCD + PNG)")
        if pcd_path is None:
            raise ValueError("--pcd 경로가 필요합니다.")
        # PNG/Intrinsics는 선택(색분할/투영 필요 시 사용)
        color_image = None
        intr = None
        if png_path:
            color_image = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if color_image is None:
                raise FileNotFoundError(f"PNG 읽기 실패: {png_path}")
        if intr_json:
            intr = load_intrinsics_json(intr_json)
            if color_image is not None and (color_image.shape[1] != intr.width or color_image.shape[0] != intr.height):
                color_image = cv2.resize(color_image, (intr.width, intr.height), interpolation=cv2.INTER_LINEAR)

        # 1) PCD 로드
        curr_pcd = o3d.io.read_point_cloud(pcd_path)
        if len(curr_pcd.points) == 0:
            raise RuntimeError(f"빈 PCD: {pcd_path}")

        # 2) 기준 실린더 생성 & ICP(y<0 절반)
        ref_pcd = create_z_axis_cylinder(
            radius=CYL_RADIUS, z_min=Z_MIN, z_max=Z_MAX, color=[1, 0, 0]
        )
        ref_pts = np.asarray(ref_pcd.points)
        aligned_pcd, T_icp = run_icp_y_negative_only(curr_pcd, ref_pcd)

        # 3) 다운샘플 & z/radius 필터 & 이상치 제거
        down = aligned_pcd.voxel_down_sample(0.01)
        pts_down = np.asarray(down.points)
        z_mask = (pts_down[:, 2] >= Z_MIN) & (pts_down[:, 2] <= Z_MAX)
        pts_clipped = pts_down[z_mask]
        radius_mask = filter_points_inside_radius(pts_clipped, (0, 0), FILTER_RADIUS)
        pts_roi = pts_clipped[radius_mask]
        if pts_roi.size == 0:
            result_json = {"success": False, "centers": []}
            print(json.dumps(result_json, ensure_ascii=False))
            return

        ab_mask, _, _ = detect_anomalies_adaptive(pts_roi, ref_pts)
        final_pts = pts_roi[ab_mask]

        # 4) 클러스터 → 중심
        mask_dummy = np.ones(len(final_pts), dtype=bool)
        centers, clusters, filtered = cluster_and_get_targets(final_pts, mask_dummy,
                                                              eps=0.037, min_samples=30)

        # (선택) 색 분할: intr/color가 있고 함수가 성공하면 색상 기준으로 다시 분할
        if color_image is not None and intr is not None and len(clusters) > 0:
            try:
                clusters = split_colorful_clusters(clusters, color_image, intr, method="mean_shift", color_thresh=1)
            except Exception as e:
                print(f"[WARN] split_colorful_clusters 실패: {e}")
        # 중심 재계산(분할이 바뀌었을 수 있음)
        if len(clusters) > 0:
            centers = [np.mean(c, axis=0) for c in clusters]
        if vis:
            try:
                visualize_pointcloud_with_garments_open3d(
                    ref_pts, pts_down, final_pts,  # ← file 모드에선 final_pts가 곧 '검출 포인트'
                    centers, filtered, clusters
                )
            except Exception as e:
                print(f"[WARN] 뷰어 실패: {e}")
        # 5) 결과 출력(JSON) — 카메라 프레임(ICP 정렬 좌표계) 기준
        if json_output:
            ori_center = centers
            # 예: 깊이 상한 보정(원 코드 따라감)
            centers.sort(key=lambda c: c[2])
            for c in centers:
                if c[2] >= 0.7:
                    c[2] = 0.7
            C_ref = np.hstack([centers, np.ones((len(centers),1))]) if centers else np.zeros((0,4))
            C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3] if len(C_ref)>0 else []

            result_json = {
                "success": len(centers) > 0,
                "log": f"origin center : {np.round(ori_center,3) if ori_center else []}, filtered center : {np.round(C_cam,3).tolist() if len(C_ref)>0 else []}",
                "centers": [c.tolist() for c in centers]
            }
            print(json.dumps(result_json, ensure_ascii=False))
        return

    # ================== 이하: 기존 실시간(wm/basket/failure) ==================
    cfg = CFG[mode]
    print(f"[INFO] mode = {mode}")

    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_cfg)

    clf, tf = (load_rgb_model() if cfg["need_clf"] else (None, None))
    label, label_col = "Press 'd' to detect", (0, 0, 0)
    cnt = 0
    try:
        while True:
            s_time = time.time()
            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()
            color_img = np.asanyarray(color.get_data())
            cv2.putText(color_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_col, 3)
            cv2.imshow("Laundry RGB Stream", color_img)
            if cnt < 30:
                cnt += 1
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            if key == ord('d') or json_output:
                if cfg["need_clf"]:
                    pred, _ = predict_rgb(clf, tf, pipeline)
                    if pred == 0:
                        print("[INFO] RGB 분류 결과 EMPTY → 탐지 스킵")

                print("[INFO] Adaptive change detection…")
                if mode == "wm":
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, (Z_MIN, Z_MAX))
                    curr_pcd = to_open3d_pcd(curr_pts)
                else:
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, (0.0, 1.0))
                    curr_pcd = to_open3d_pcd(curr_pts)

                if mode == "wm":
                    center = estimate_center_from_pcd(curr_pcd)
                    ref_pcd = create_z_axis_cylinder(radius=CYL_RADIUS, z_min=Z_MIN, z_max=Z_MAX, color=[1, 0, 0])
                    ref_pts = np.asarray(ref_pcd.points)

                    aligned_pcd, T_icp = run_icp_y_negative_only(curr_pcd, ref_pcd)

                    down = aligned_pcd.voxel_down_sample(0.01)
                    pts_down = np.asarray(down.points)
                    zmin, zmax = cfg["z_range"]
                    z_mask = (pts_down[:, 2] >= zmin) & (pts_down[:, 2] <= zmax)
                    pts_clipped = pts_down[z_mask]
                    radius_mask = filter_points_inside_radius(pts_clipped, (0, 0), FILTER_RADIUS)
                    pts_roi = pts_clipped[radius_mask]

                    ab_mask, _, _ = detect_anomalies_adaptive(pts_roi, ref_pts)
                    pts_roi = pts_roi[ab_mask]
                    _, _, wings = cluster_and_get_targets(pts_roi, np.ones(len(pts_roi), dtype=bool),
                                                          eps=cfg["eps"], min_samples=cfg["min_samples"])
                    rotated_wings = []
                    for wing in wings:
                        rotated_wings.extend([rotate_cluster(wing, (0,0), 120), rotate_cluster(wing, (0,0), 240)])
                    if rotated_wings or wings:
                        all_wing_pts = np.vstack(rotated_wings + wings)
                        print("wing detected")
                    else:
                        all_wing_pts = np.empty((0, 3)); print("no detected wings")

                    def is_close_to_wing(pt):
                        diffs = all_wing_pts - pt
                        xy_dists = np.linalg.norm(diffs[:, :2], axis=1)
                        z_dists = np.abs(diffs[:, 2])
                        return np.any((xy_dists < 0.02) & (z_dists < 0.15))

                    mask_keep = np.array([not is_close_to_wing(p) for p in pts_roi])
                    final_pts = pts_roi[mask_keep]

                elif mode == "basket":
                    ref_pcd = o3d.io.read_point_cloud(cfg["ref_pcd"])
                    pts = np.asarray(ref_pcd.points)
                    ref_pcd.points = o3d.utility.Vector3dVector(pts[pts[:, 2] <= cfg["z_crop"][1]])
                    ref_pts = np.asarray(ref_pcd.points)
                    aligned_pts, T_icp = align_icp(np.asarray(curr_pcd.points), ref_pts)
                    pcd_down = to_open3d_pcd(aligned_pts).voxel_down_sample(0.01)
                    pts_down = np.asarray(pcd_down.points)
                    zmin, zmax = cfg["z_range"]
                    final_pts = pts_down[(pts_down[:, 2] >= zmin) & (pts_down[:, 2] <= zmax)]

                elif mode == "failure":
                    ref_pcd = o3d.io.read_point_cloud(cfg["ref_pcd"])
                    pts = np.asarray(ref_pcd.points)
                    ref_pcd.points = o3d.utility.Vector3dVector(pts[pts[:, 2] <= cfg["z_crop"][1]])
                    ref_pts = np.asarray(ref_pcd.points)
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, depth_crop=cfg["z_range"])
                    curr_pcd = to_open3d_pcd(curr_pts)
                    aligned_pts, _ = align_icp(np.asarray(curr_pcd.points), ref_pts)
                    pcd_down = to_open3d_pcd(aligned_pts).voxel_down_sample(0.01)
                    pts_down = np.asarray(pcd_down.points)
                    final_pts = pts_down[(pts_down[:, 2] >= cfg["z_range"][0]) & (pts_down[:, 2] <= cfg["z_range"][1])]

                mask, _, _ = detect_anomalies_adaptive(final_pts, ref_pts)
                centers, clusters, filtered = cluster_and_get_targets(final_pts, mask,
                                                                      eps=cfg["eps"], min_samples=cfg["min_samples"])
                clusters = split_colorful_clusters(clusters, color_image, intr, method="mean_shift", color_thresh=1)
                centers = [np.mean(c, axis=0) for c in clusters]

                if clusters and len(clusters) > 0:
                    print("do color lab")
                    same_clusters, diff_clusters = split_clusters_by_color_lab(
                        clusters, final_pts, color_image, intr
                    )
                else:
                    same_clusters, diff_clusters = [], []

                if centers:
                    print(f" - 탐지된 클러스터 수: {len(centers)}")
                    for i, c in enumerate(centers):
                        print(f"   grasp {i}: {np.round(c, 3)}")
                    label, label_col = f"CLOTH", (0, 0, 255)
                else:
                    print(" - 의류 클러스터 미발견")
                    label, label_col = "EMPTY", (0, 255, 0)

                if json_output:
                    print(f"before cali : {centers}")
                    ori_center = []
                    if mode == "basket":
                        ori_center = centers
                        new_center = [c for c in centers if -0.2 < c[0] < 0.2 and -0.145 < c[1] < 0.145]
                        for c in new_center:
                            c[2] += 0.02
                        new_center.sort(key=lambda c: np.linalg.norm([c[0], c[1]]))
                        centers = new_center
                        C_ref = np.hstack([centers, np.ones((len(centers),1))])
                        C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3]
                        result_json = {"success": len(C_cam) > 0,
                                       "log": f"origin center : {ori_center}, filtered center : {C_cam}",
                                       "centers": [c.tolist() for c in C_cam]}
                        print(json.dumps(result_json))
                    elif mode == "wm":
                        ori_center = centers
                        centers.sort(key = lambda c: c[2])
                        for c in centers:
                            if c[2] >= 0.7:
                                c[2] = 0.7
                        print(f"after cali {centers}")
                        C_ref = np.hstack([centers, np.ones((len(centers),1))])
                        C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3]
                        result_json = {"success": len(centers) > 0,
                                       "log": f"origin center : {ori_center}, filtered center : {C_cam}",
                                       "centers": [c.tolist() for c in centers]}
                        print(json.dumps(result_json))
                    return

                e_time = time.time()
                print(f"detect_takes : {e_time-s_time}")
                visualize_pointcloud_with_garments_open3d(
                    ref_pts, pts_down, final_pts[mask],
                    centers, filtered, same_clusters + diff_clusters)

    finally:
        if mode != "file":
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wm", "basket", "failure", "file"], required=True)
    ap.add_argument("--json", action="store_true", help="JSON 포맷으로 결과 출력")
    ap.add_argument("--vis", action="store_true", help="[file] Open3D로 PCD/결과 시각화")
    # file 모드 전용 인자
    ap.add_argument("--pcd", type=str, default=None, help="[file] 단일 PCD 경로(.pcd/.ply)")
    ap.add_argument("--png", type=str, default=None, help="[file] 단일 PNG/JPG 경로")
    ap.add_argument("--intr_json", type=str, default=None, help="[file] intrinsics JSON(fx,fy,ppx,ppy,width,height)")

    args = ap.parse_args()
    run(args.mode, args.json, pcd_path=args.pcd, png_path=args.png, intr_json=args.intr_json, vis=args.vis)