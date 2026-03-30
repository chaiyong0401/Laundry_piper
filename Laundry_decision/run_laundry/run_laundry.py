'''
python run_laundry.py   --mode wm   --rgb_ckpt /home/cheon/Documents/gd_0808/models/best.pt   --rgb_thr 0.60 --rgb_low_thr 0.45 --json

'''
import argparse, time, cv2, numpy as np, pyrealsense2 as rs, open3d as o3d
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import sys, json, time


from detector.rgb_classifier   import RGBOnlyClassifier
from detector.utils            import (get_current_pointcloud, load_and_crop_pcd,
                                       align_icp, to_open3d_pcd, split_clusters_by_color_lab)
from detector.zscore_detector  import (detect_anomalies_adaptive,
                                       cluster_and_get_targets,
                                       visualize_pointcloud_with_garments_open3d, split_colorful_clusters)
# wm global var
Z_MIN, Z_MAX = 0.47, 0.90
CYL_RADIUS = 0.29
FILTER_RADIUS = 0.285

CFG = {
    "wm": {
        "ref_pcd"      : "reference/washer.pcd",
        "z_crop"       : (0.6, 1.2),
        "z_range"      : (Z_MIN, Z_MAX),
        "need_clf"     : True,
        "eps"          : 0.037, "min_samples": 40 
        # "eps"          : 0.05, "min_samples": 30 # debug
    },
    "basket": {
        "ref_pcd"      : "reference/basket.pcd",
        "z_crop"       : (0.0, 1.0),
        "z_range"      : (0.10, 0.60),
        "need_clf"     : False,
        "eps"          : 0.037, "min_samples": 30
    },
    # "failure": {
    #     "ref_pcd": "reference/failure_empty.pcd",     
    #     "z_crop": (0.0, 0.5),                         
    #     "z_range": (0.0, 0.5),
    #     "need_clf": False,
    #     "eps": 0.037, "min_samples": 30
    # }

}

# run_laundry.py 어딘가(함수 밖, 전역) 붙여넣기
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _rgb_eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _create_backbone(model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    if model_name == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif model_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError(f"unknown model: {model_name}")

class _RGBGate:
    def __init__(self, ckpt_path: str, device_str: str = "cpu", ema_decay: float = 0.7):
        import os, json
        assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
        self.device = torch.device(device_str)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[_RGBGate] CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        self.tfm = _rgb_eval_transform()
        self.ema_decay = ema_decay
        self.ema_prob = None
        self.ts = None
        self.model = None
        self.idx_to_class = {0: "empty", 1: "cloth"}
        self.cloth_idx = 1

        # 1) TorchScript 우선 로드 시도
        try:
            self.ts = torch.jit.load(ckpt_path, map_location=self.device)
            self.ts.eval()
            # (선택) 옆에 .json 메타가 있으면 라벨 매핑 사용
            meta_path = ckpt_path + ".json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "idx_to_class" in meta:
                    self.idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
                    # cloth 인덱스 갱신
                    self.cloth_idx = [k for k, v in self.idx_to_class.items() if v == "cloth"]
                    self.cloth_idx = self.cloth_idx[0] if len(self.cloth_idx) else 1
            print(f"[rgb-gate] TorchScript loaded: {ckpt_path}")
            return
        except Exception:
            self.ts = None  # fallback to state_dict

        # 2) 학습 체크포인트(state_dict) 로드
        ckpt = torch.load(ckpt_path, map_location=self.device)
        args = ckpt.get("args", {})
        model_name = args.get("model", "resnet18")
        self.model = _create_backbone(model_name, num_classes=2).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"], strict=False)

        cls2idx = ckpt.get("class_to_idx", {"empty": 0, "cloth": 1})
        self.idx_to_class = {v: k for k, v in cls2idx.items()}
        self.cloth_idx = [k for k, v in self.idx_to_class.items() if v == "cloth"]
        self.cloth_idx = self.cloth_idx[0] if len(self.cloth_idx) else 1
        print(f"[rgb-gate] state_dict loaded: {ckpt_path}, cloth_idx={self.cloth_idx}")

    @torch.inference_mode()
    def _predict(self, bgr_img):  # np.uint8 HxWx3 (OpenCV/RealSense BGR 가정)
        import numpy as np
        assert bgr_img is not None and bgr_img.ndim == 3 and bgr_img.shape[2] == 3
        rgb = bgr_img[..., ::-1]                 # BGR → RGB
        x = self.tfm(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self.ts is not None:
            logits = self.ts(x)
        else:
            logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred_idx = int(prob.argmax().item())
        cloth_prob = float(prob[self.cloth_idx].item()) if 0 <= self.cloth_idx < prob.numel() else float(prob.max().item())
        return pred_idx, cloth_prob

    @torch.inference_mode()
    def decide(self, bgr_img, hi_thr: float = 0.60, lo_thr: float = 0.45):
        pred_idx, p = self._predict(bgr_img)
        self.ema_prob = p if self.ema_prob is None else (self.ema_decay * self.ema_prob + (1 - self.ema_decay) * p)
        if self.ema_prob >= hi_thr:
            decision = "CLOTH"
        elif self.ema_prob < lo_thr:
            decision = "EMPTY"
        else:
            decision = "CLOTH"  # 보수적(빈 오판 방지)
        return {
            "pred_label": self.idx_to_class.get(pred_idx, str(pred_idx)),
            "cloth_prob": float(p),
            "ema_prob": float(self.ema_prob),
            "decision": decision
        }


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
    """
    원점 기준 Z축 방향 실린더 생성 (기준축이 z축임)
    """
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


def run_icp_custom(curr_pcd, ref_pcd):
    ref_pts = np.asarray(ref_pcd.points)
    curr_pts = np.asarray(curr_pcd.points)

    ref_mask = ref_pts[:,1] < 0.0
    curr_mask = curr_pts[:,1] < 0.0

    ref_half = o3d.geometry.PointCloud()
    curr_half = o3d.geometry.PointCloud()
    ref_half.points = o3d.utility.Vector3dVector(ref_pts[ref_mask])
    curr_half.points = o3d.utility.Vector3dVector(curr_pts[curr_mask])

    reg = o3d.pipelines.registration.registration_icp(
        curr_half.voxel_down_sample(0.005),
        ref_half.voxel_down_sample(0.005),
        0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    curr_pcd.transform(reg.transformation)
    return curr_pcd, reg.transformation

def run_icp_y_negative_only(curr_pcd, ref_pcd):
    """
    curr_pcd와 ref_pcd의 y<0 부분만 추출해 ICP 정합을 수행함
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
    R = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    T = np.array([0.05, 0, -0.05])
    return R @ p_cam + T 


def run(mode, json_output, show, device="cpu", rgb_ckpt="models/best.pt"):
    s_time = time.time()
    cfg = CFG.get(mode)
    if not cfg:
        print(f"Unknown mode: {mode}")
        return

    # 1. model
    ckpt_path = rgb_ckpt
    rgb_gate  = _RGBGate(ckpt_path, device_str=device)

    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # --- 카메라 초기화 안정화 로직 ---
    # 주의: 반드시 각 시도마다 pipeline 객체를 새로 생성해야 함.
    # 이전 시도에서 start() 중 타임아웃이 발생해도 기존 pipeline 객체는
    # 내부적으로 카메라 핸들을 점유하고 있어, 같은 객체로 재시도 시
    # "UVC device is already opened" 에러가 발생한다.
    max_retries = 5
    pipeline = None
    last_exc = None
    for attempt in range(max_retries):
        try:
            pipeline = rs.pipeline()  # 매번 새 파이프라인 객체 생성
            profile = pipeline.start(rs_cfg)
            
            # --- [참고용: 카메라 노출(AE) 고정 세팅 (임시 비활성화)] ---
            # device = profile.get_device()
            # color_sensor = None
            # for sensor in device.query_sensors():
            #     if sensor.is_color_sensor():
            #         color_sensor = sensor
            #         try:
            #             # 초기에는 자동 노출을 켜서 주변 밝기에 적응하게 둠
            #             sensor.set_option(rs.option.enable_auto_exposure, 1)
            #             sensor.set_option(rs.option.enable_auto_white_balance, 1)
            #         except: pass
            #     elif sensor.is_depth_sensor():
            #         try:
            #             sensor.set_option(rs.option.enable_auto_exposure, 0)
            #         except: pass
            # ----------------------------------------------------
            
            print(f"[INFO] RealSense Pipeline Started (Attempt {attempt+1})")
            break
        except Exception as e:
            last_exc = e
            print(f"[ERROR] Pipeline start failed (Attempt {attempt+1}): {e}")
            # 기존 파이프라인 객체의 핸들 해제 시도
            try:
                pipeline.stop()
            except:
                pass
            pipeline = None

            if attempt < max_retries - 1:
                print("[INFO] Releasing handle and waiting before retry...")
                try:
                    # librealsense의 context를 통해 하드웨어 리셋 시도
                    ctx = rs.context()
                    devices = ctx.query_devices()
                    for dev in devices:
                        dev.hardware_reset()
                    time.sleep(4.0)  # 리셋 후 장치 재연결 대기 (충분히 기다림)
                except:
                    time.sleep(2.0)
            else:
                print("[FATAL] Camera initialization failed after all attempts.")
                raise last_exc

    if pipeline is None:
        raise RuntimeError("RealSense pipeline could not be initialized.")
    # ------------------------------------

    # main() 시작 직후나, 카메라/세션 초기화 직후
    rgb_gate = None
    if args.rgb_ckpt:
        try:
            rgb_gate = _RGBGate(args.rgb_ckpt, device_str="cuda")
        except Exception as e:
            print(f"[rgb-gate][warn] load failed: {e}")
            rgb_gate = None
    cnt =0
    try:
        while True:
            s_time = time.time()
            frames = pipeline.wait_for_frames()
            color  = frames.get_color_frame()

            if not color:
                print("[WARN] ❌ Color frame is None!")
                continue

            color_img = np.asanyarray(color.get_data())
            label = ''; label_col = (0,1,0)
            cv2.putText(color_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_col, 3)    # debug
            if show:
                cv2.imshow("Laundry RGB Stream", color_img)
            if cnt < 30:
                cnt +=1 
                # --- [참고용: 30프레임 대기 후 노출 고정 (임시 비활성화)] ---
                # if cnt == 1:
                #     print("[INFO] Warming up camera (Auto-Exposure ON for 30 frames)...")
                # if cnt == 30 and 'color_sensor' in locals() and color_sensor is not None:
                #     try:
                #         color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                #         color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                #         e = color_sensor.get_option(rs.option.exposure)
                #         g = color_sensor.get_option(rs.option.gain)
                #         print(f"[INFO] Camera warmed up! Locked AE -> Exposure:{e}, Gain:{g}")
                #     except Exception as exc:
                #         print(f"[WARN] Failed to lock AE: {exc}")
                # --------------------------------------------------------
                continue
            
            key = 0
            if show:
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            if key == ord('d') or json_output:  
                
#                print(f'cfg : {cfg["need_clf"]}')

                if cfg["need_clf"]:
                    # color_image: np.uint8 (H,W,3) BGR 프레임 (RealSense/ OpenCV 기본)
                    if mode == "wm" and rgb_gate is not None and color_img is not None:
                        g = rgb_gate.decide(color_img, hi_thr=args.rgb_thr, lo_thr=args.rgb_low_thr)
                        # 매우 EMPTY에 가깝다고 확신되면 바로 스킵
                        print(g["decision"])
                        if g["decision"] == "EMPTY":
                            print("Decision is EMPTY")
                        if g["decision"] == "EMPTY" and g["ema_prob"] < (args.rgb_thr - 0.05):
                            log = f"rgb-empty-skip p={g['cloth_prob']:.2f} ema={g['ema_prob']:.2f} thr={args.rgb_thr:.2f}"
                            print("[wm] " + log)
                            res = {"success": False, "centers": [], "log": log, "reason": "rgb_gate_empty"}
                            if json_output:
                                print(json.dumps(res))
                            return res
                        else:
                            log = f"rgb-cloth p={g['cloth_prob']:.2f} ema={g['ema_prob']:.2f} thr={args.rgb_thr:.2f}"
                            print("[wm] " + log)
                    
                    
                print("[INFO] Adaptive change detection…")
                if mode == "wm":
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, (Z_MIN, Z_MAX))
                    curr_pcd = to_open3d_pcd(curr_pts)
                else:
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, (0.0, 1.0))
                    curr_pcd = to_open3d_pcd(curr_pts)
                    
                    
                if mode == "wm":
                    center = estimate_center_from_pcd(curr_pcd)
                    #cx, cy = center[0], center[1]
                    #print(f"center x,y : {cx}, {cy}")
                    ref_pcd = create_z_axis_cylinder(
                        #center_xy=(cx, cy),
                        radius=CYL_RADIUS,
                        z_min=Z_MIN,
                        z_max=Z_MAX,
                        color=[1, 0, 0]
                    )
                    ref_pts = np.asarray(ref_pcd.points)
                    
                    aligned_pcd, T_icp = run_icp_y_negative_only(curr_pcd, ref_pcd)

                    down = aligned_pcd.voxel_down_sample(0.01)
                    pts_down = np.asarray(down.points)
                    zmin, zmax = cfg["z_range"]
                    z_mask = (pts_down[:, 2] >= zmin) & (pts_down[:, 2] <= zmax)
                    pts_clipped = pts_down[z_mask]

                    # 날개 탐기 위한 반지름 필터링
                    radius_mask = filter_points_inside_radius(pts_clipped, (0, 0), FILTER_RADIUS)
                    pts_roi = pts_clipped[radius_mask]  # (N, 3)

                    # 날개 탐지 위한 Adaptive threshold 기반 이상치 필터링
                    ab_mask, _, _ = detect_anomalies_adaptive(pts_roi, ref_pts)
                    pts_roi = pts_roi[ab_mask]  
                    print("pts_roi:", len(pts_roi))
                    print("ref_pts:", len(ref_pts))
                    # 날개 탐지
                    _, _, wings = cluster_and_get_targets(pts_roi, np.ones(len(pts_roi), dtype=bool), eps=cfg["eps"], min_samples=cfg["min_samples"])
                    rotated_wings = []
                    for wing in wings:
                        rotated_wings.extend([
                            rotate_cluster(wing, (0,0), 120),
                            rotate_cluster(wing, (0,0), 240)
                        ])

                    # 날개 근처 제거 마스크 적용
                    if rotated_wings or wings:
                        all_wing_pts = np.vstack(rotated_wings + wings)
                        print("wing detected")
                    else:
                        all_wing_pts = np.empty((0, 3)) 
                        print("no detected wings")
                        
                    def is_close_to_wing(pt):
                        diffs = all_wing_pts - pt  # shape: (N, 3)
                        xy_dists = np.linalg.norm(diffs[:, :2], axis=1)  # XY 평면 거리
                        z_dists = np.abs(diffs[:, 2])                    # Z축 거리
                        close_mask = (xy_dists < 0.02) & (z_dists < 0.15)
                        return np.any(close_mask)

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

                    # 현재 프레임 포인트클라우드 수집
                    curr_pts, color_image, intr = get_current_pointcloud(pipeline, depth_crop=cfg["z_range"])
                    curr_pcd = to_open3d_pcd(curr_pts)

                    # ICP 정합
                    aligned_pts, _ = align_icp(np.asarray(curr_pcd.points), ref_pts)

                    # 다운샘플 & z 필터링
                    pcd_down = to_open3d_pcd(aligned_pts).voxel_down_sample(0.01)
                    pts_down = np.asarray(pcd_down.points)
                    final_pts = pts_down[(pts_down[:, 2] >= cfg["z_range"][0]) & (pts_down[:, 2] <= cfg["z_range"][1])]

                    

                mask, _, _ = detect_anomalies_adaptive(final_pts, ref_pts)
                centers, clusters, filtered = cluster_and_get_targets(final_pts, mask, eps=cfg["eps"], min_samples=cfg["min_samples"])
                clusters = split_colorful_clusters(clusters, color_image, intr, method="mean_shift", color_thresh=1) # debug
                # clusters = split_colorful_clusters(clusters, color_image, intr, method="mean_shift", color_thresh=10) # debug
                
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
                        # 1) 범위 
                        new_center = [
                            c for c in centers
                            if -0.2 < c[0] < 0.2 and -0.145 < c[1] < 0.145
                        ]
                        for c in new_center:
                            c[2] +=0.02
                            print(c[2])
                        
                        # 2) XY 거리 기준 정렬
                        new_center.sort(key=lambda c: np.linalg.norm([c[0], c[1]]))
                        centers = new_center
                        C_ref = np.hstack([centers, np.ones((len(centers),1))])  # (N,4)
                        C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3]   
                        
                        
                        result_json = {
                            "success": len(C_cam) > 0,
                            "log": f"origin center : {ori_center}, filtered center : {C_cam}",
                            "centers": [c.tolist() for c in C_cam]
                        }
                        
                        print(json.dumps(result_json))
                        
                    elif mode == "wm":
                        ori_center = centers
                        centers.sort(key=lambda c: np.linalg.norm([c[0], c[1]]))
                        
                        new_center = []
                        for c in centers:
                            if c[2] >= 0.7:
                                c[2] = 0.7
                            # if not c[1] > 0.24:
                            #     c[1] +=0.04
                            if c[1] > 0.29:
                                c[1] = 0.29
                            print(c)
                            
                        print(f"after cali {centers}")

                        if len(centers) == 0:
                            result_json = {
                                "success": False,
                                "log": f"origin center : {ori_center}, no clusters detected",
                                "centers": []
                            }
                            print(json.dumps(result_json))
                            return
                        
                        
                        C_ref = np.hstack([centers, np.ones((len(centers),1))])
                        C_cam = (np.linalg.inv(T_icp) @ C_ref.T).T[:, :3]

                        result_json = {
                            "success": True,
                            "log": f"origin center : {ori_center}, filtered center : {C_cam}",
                            "centers": [c.tolist() for c in C_cam]
                        }

                        print(json.dumps(result_json))
                    
                    
                    
                # End of if json_output block (indentation level adjusted below)
                
                e_time = time.time()
                print(f"detect_takes : {e_time-s_time}")
                
                # debug
                if show:
                    visualize_pointcloud_with_garments_open3d(
                        ref_pts, pts_down, final_pts[mask],
                        centers, filtered, same_clusters + diff_clusters)
                
                return


    finally:
        pipeline.stop()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wm", "basket", "failure"], required=True)
    ap.add_argument("--json", action="store_true", help="JSON 포맷으로 결과 출력")
    ap.add_argument("--rgb_ckpt", type=str, default="models/best.pt",
                    help="RGB empty/cloth 분류 ckpt (.pt/.pth 또는 TorchScript)")
    ap.add_argument("--rgb_thr", type=float, default=0.8,
                        help="cloth 확률 상한 임계값")
    ap.add_argument("--rgb_low_thr", type=float, default=0.7,
                        help="cloth 확률 하한 임계값(히스테리시스)")
    ap.add_argument("--show", action="store_true", help="시각화 창 표시")
    ap.add_argument("--device", type=str, default="cpu", help="device for inference (cuda or cpu)")
    args = ap.parse_args()
    run(args.mode, args.json, args.show, args.device, args.rgb_ckpt)
