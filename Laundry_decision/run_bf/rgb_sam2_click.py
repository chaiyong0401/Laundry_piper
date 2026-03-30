'''
python rgb_sam2_click.py \
  --in_root /home/cheon/Documents/run_bf/dataset \
  --out_root /home/cheon/Documents/run_bf/dataset_rgb_sam2/wm \
  --sam2_ckpt /home/cheon/Documents/run_bf/checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
  --device cuda \
  --enable_zmask --zmax 0.65 \
  --save_diff_cur \
  --ref_match cached --match_down 4 --ref_cache /home/cheon/Documents/run_failure/ref_cache_basket.json \
  --align affine \
  --depth_suffix _depth --rgb_suffix _rgb --strip_suffix --rgb_exts .png,.jpg \
  --domain dataset_wm/cloth \
  --empty_domain dataset_wm/empty

'''

# python rgb_sam2_click.py \
#   --in_root   /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset \
#   --out_root  /home/dyros/mcy_ws/Laundry_decision/run_bf/dataset_rgb_sam2/min_failure \
#   --sam2_ckpt /home/dyros/mcy_ws/Laundry_decision/run_bf/checkpoints/sam2.1_hiera_base_plus.pt \
#   --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
#   --device    cuda \
#   --enable_zmask --zmax 0.65 \
#   --save_diff_cur \
#   --ref_match best_ecc --match_down 4 \
#   --align affine \
#   --depth_suffix _depth --rgb_suffix _rgb --strip_suffix \
#   --rgb_exts .png,.jpg,.jpeg \
#   --domain dataset_min_failure/cloth \
#   --empty_domain dataset_min_failure/empty



# sam2.1_hiera_b+.yaml
# sam2.1_hiera_s.yaml
# sam2.1_hiera_t.yaml
# rgb_sam2_click.py


import os, glob, argparse, json, cv2, numpy as np
import torch, time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

WIN = "rgb-sam2-click-fast"

def imread_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        d = img.astype(np.float32)
        if d.max() > 50: d /= 1000.0  # mm->m
        return d
    if img.dtype in (np.float32, np.float64):
        return img.astype(np.float32)
    return img.astype(np.float32)/255.0

def ecc_align(src, dst):
    src_n = src.astype(np.float32).copy()
    dst_n = dst.astype(np.float32).copy()
    for a in (src_n, dst_n):
        p1, p99 = np.percentile(a, 1), np.percentile(a, 99)
        a[:] = np.clip((a - p1) / max(1e-6, p99 - p1), 0, 1)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(
            dst_n, src_n, M, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        )
    except cv2.error:
        pass
    h, w = dst.shape[:2]
    return cv2.warpAffine(
        src, M, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

def find_match(stem, dirpath, exts, depth_suffix, rgb_suffix, strip_suffix):
    cands = [stem, stem.replace(depth_suffix, rgb_suffix)]
    if strip_suffix:
        cands.append(stem.replace(depth_suffix, ''))
    cands.append(stem.replace(rgb_suffix, ''))
    for s in cands:
        for ex in exts:
            p = os.path.join(dirpath, s + ex)
            if os.path.exists(p):
                return p
    return None

class App:
    def __init__(self, args):
        self.args = args
        self.device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # --- SAM2 ---
        model = build_sam2(args.sam2_cfg, args.sam2_ckpt)
        self.predictor = SAM2ImagePredictor(model)
        self.predictor.model.to(self.device)

        # --- Domain-aware IO dirs ---
        self.domain = args.domain

        # (a) in_root가 이미 rgb/depth_img 구조인지 확인
        cand_rgb   = os.path.join(args.in_root, args.rgb_dirname)
        cand_depth = os.path.join(args.in_root, args.depth_dirname)
        if os.path.isdir(cand_rgb) and os.path.isdir(cand_depth):
            domain_root = args.in_root
        else:
            # (b) in_root/<domain> 구조
            domain_root = os.path.join(args.in_root, self.domain)

        self.rgb_dir   = os.path.join(domain_root, args.rgb_dirname)
        self.depth_dir = os.path.join(domain_root, args.depth_dirname)

        # --- Empty domain root 결정 ---
        empty_root = None
        if args.empty_dir:
            empty_root = args.empty_dir  # 사용자가 직접 지정
        else:
            empty_domain = args.empty_domain or f"{self.domain}"
            cand1 = os.path.join(args.in_root, empty_domain)
            cand2 = os.path.join(os.path.dirname(domain_root), empty_domain)
            for c in (cand1, cand2):
                if os.path.isdir(c):
                    empty_root = c
                    break

        self.empty_depth_dir = os.path.join(empty_root, args.depth_dirname) if empty_root else None

        # 출력 준비
        self.samples_root = os.path.join(args.out_root, 'samples')
        os.makedirs(self.samples_root, exist_ok=True)

        # 입력 depth 목록
        self.depth_dir = os.path.join(self.depth_dir)
        self.depths = sorted(glob.glob(os.path.join(self.depth_dir, '*.png')))
        
        if len(self.depths) == 0:
            raise SystemExit(f"depth 이미지가 없습니다: {self.depth_dir}")

        # RGB 확장자
        self.exts = [e.strip() for e in args.rgb_exts.split(',') if e.strip()]

        # Empty preload (저해상도 캐시)
        self.empty_paths = sorted(glob.glob(os.path.join(self.empty_depth_dir, '*.png'))) if self.empty_depth_dir else []
        if args.save_diff_cur and len(self.empty_paths) == 0:
            raise SystemExit("`--save_diff_cur` 사용 시 empty depth 폴더가 필요합니다.")

        self.empty_small = []
        self._H = self._W = None
        if self.empty_paths:
            d0 = imread_depth(self.empty_paths[0]); self._H, self._W = d0.shape
            sh, sw = max(1, self._H // args.match_down), max(1, self._W // args.match_down)
            for p in self.empty_paths:
                e = imread_depth(p)
                e_s = cv2.resize(e, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                self.empty_small.append(e_s)

        # Ref cache (매칭 결과 저장/불러오기)
        self.ref_cache = {}
        if args.ref_cache and os.path.exists(args.ref_cache):
            try:
                self.ref_cache = json.load(open(args.ref_cache, 'r'))
            except Exception:
                pass

        # State
        self.idx = 0
        self.mask = None
        self.pos = []; self.neg = []
        self._edges = None; self._edges_from = None
        self._cur_rgb_bgr = None; self._cur_depth = None
        self._cur_stem = None
        self._d_diff_ready = False
        self._d_ref_al = None
        self._d_diff = None
        self._ref_path = None
        self.zmask = None

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.loop()

    def current_paths(self):
        cur_d = self.depths[self.idx]
        stem = os.path.splitext(os.path.basename(cur_d))[0]
        rgb = find_match(
            stem, self.rgb_dir, self.exts,
            self.args.depth_suffix, self.args.rgb_suffix, self.args.strip_suffix
        )
        return stem, cur_d, rgb

    def best_empty_for(self, d_cur):
        # 전략 선택
        if self.args.ref_match == 'first' or len(self.empty_paths) == 0:
            return self.empty_paths[0] if self.empty_paths else None
        if self.args.ref_match == 'idx':
            idx = max(0, min(self.args.ref_idx, len(self.empty_paths) - 1))
            return self.empty_paths[idx]

        # cached or best_ecc
        stem = self._cur_stem
        if self.args.ref_match in ('cached', 'best_ecc'):
            if stem in self.ref_cache:
                idx = self.ref_cache[stem]
                return self.empty_paths[idx]
            # 빠른 매칭(저해상도, MAE)
            sh, sw = self.empty_small[0].shape
            d_s = cv2.resize(d_cur, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            best_i, best_sc = 0, 1e9
            for i, e_s in enumerate(self.empty_small):
                sc = np.mean(np.abs(d_s - e_s))
                if sc < best_sc:
                    best_sc, best_i = sc, i
            if self.args.ref_match == 'cached':
                self.ref_cache[stem] = best_i
            return self.empty_paths[best_i]

    def build_bg(self):
        stem, cur_d_path, rgb_path = self.current_paths()
        self._cur_stem = stem
        d_cur = imread_depth(cur_d_path)
        self._cur_depth = d_cur
        self.zmask = ((d_cur > 0) & (d_cur <= self.args.zmax)).astype(np.uint8) * 255 if self.args.enable_zmask else None

        # RGB 로드 (BGR)
        if rgb_path and os.path.exists(rgb_path):
            self._cur_rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        else:
            # fallback: depth colormap
            vals = d_cur[(d_cur > 0) & np.isfinite(d_cur)]
            if vals.size < 50:
                lo, hi = 0.0, self.args.zmax
            else:
                lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
                hi = max(hi, lo + 1e-3)
            dn = np.clip((d_cur - lo) / max(1e-6, hi - lo), 0, 1)
            self._cur_rgb_bgr = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

        # diff 준비 플래그만 리셋 (계산은 save 때)
        self._d_diff_ready = False
        self._d_ref_al = None
        self._d_diff = None
        self._ref_path = None
        self.mask = None; self.pos.clear(); self.neg.clear()
        self._edges = None; self._edges_from = None

    def ensure_diff_ready(self):
        if self._d_diff_ready or not self.args.save_diff_cur:
            return
        ref_path = self.best_empty_for(self._cur_depth)
        if ref_path is None:
            raise RuntimeError("empty ref not found while saving diff/cur")
        d_ref = imread_depth(ref_path)
        if self.args.align == 'none':
            d_ref_al = d_ref
        else:
            d_ref_al = ecc_align(d_ref, self._cur_depth)
        self._d_ref_al = d_ref_al
        self._d_diff = np.abs(self._cur_depth - d_ref_al)
        self._ref_path = ref_path
        self._d_diff_ready = True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pos.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.neg.append((x, y))

    def run_sam2(self):
        img_rgb = cv2.cvtColor(self._cur_rgb_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            if self.device.startswith('cuda'):
                ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            else:
                ctx = torch.cpu.amp.autocast(enabled=False)
            with ctx:
                self.predictor.set_image(img_rgb)
                pm, pl = None, None
                if len(self.pos) + len(self.neg) > 0:
                    pts, lbs = [], []
                    if self.pos: pts += self.pos; lbs += [1] * len(self.pos)
                    if self.neg: pts += self.neg; lbs += [0] * len(self.neg)
                    pm = np.array(pts, np.float32); pl = np.array(lbs, np.int32)
                masks, scores, _ = self.predictor.predict(point_coords=pm, point_labels=pl, multimask_output=True)
        best = masks[np.argmax(scores)].astype(np.uint8) * 255
        if self.zmask is not None:
            best = (best & self.zmask)
        self.mask = best if self.mask is None else cv2.bitwise_or(self.mask, best)
        self._edges = None  # invalidate

    def _get_edges(self, M):
        key = (id(M), M.shape, int(M.sum()))
        if self._edges_from != key:
            self._edges = cv2.Canny(M, 50, 150)
            self._edges_from = key
        return self._edges

    def save(self):
        self.ensure_diff_ready()  # lazy 계산
        stem, cur_d_path, rgb_path = self.current_paths()
        out_dir = os.path.join(self.samples_root, stem)
        os.makedirs(out_dir, exist_ok=True)

        M = self.mask if self.mask is not None else np.zeros(self._cur_rgb_bgr.shape[:2], np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'mask.png'), M)

        if self.args.save_diff_cur:
            np.save(os.path.join(out_dir, 'diff.npy'), self._d_diff.astype(np.float32))
            np.save(os.path.join(out_dir, 'cur.npy'),  self._cur_depth.astype(np.float32))

        # RGB도 함께 저장 (학습 호환)
        rgb_save = self._cur_rgb_bgr[:, :, ::-1]  # BGR->RGB
        cv2.imwrite(os.path.join(out_dir, 'rgb.png'), rgb_save[:, :, ::-1])  # 다시 BGR로 저장(표준)

        dbg = self._cur_rgb_bgr.copy()
        edges = self._get_edges(M)
        dbg[edges > 0] = (0, 0, 255)
        cv2.imwrite(os.path.join(out_dir, 'dbg_overlay.png'), dbg)

        meta = dict(
            clicks=dict(pos=self.pos, neg=self.neg),
            rgb_path=rgb_path, depth_path=cur_d_path,
            zmax=float(self.args.zmax), enable_zmask=bool(self.args.enable_zmask)
        )
        if self.args.save_diff_cur:
            meta['ref_path'] = self._ref_path
        json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)
        print("Saved:", out_dir)

        # ref_cache 저장
        if self.args.ref_cache:
            try:
                json.dump(self.ref_cache, open(self.args.ref_cache, 'w'), indent=2)
            except Exception:
                pass

    def loop(self):
        while 0 <= self.idx < len(self.depths):
            self.build_bg()
            while True:
                canvas = self._cur_rgb_bgr.copy()
                for x, y in self.pos:
                    cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
                for x, y in self.neg:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
                if self.mask is not None:
                    edges = self._get_edges(self.mask)
                    canvas[edges > 0] = (0, 0, 255)

                cv2.imshow(WIN, canvas)
                k = cv2.waitKey(20) & 0xFF
                if k == ord('e'):  # run SAM
                    self.run_sam2()
                elif k in (ord('m'), ord('M')):  # clear mask
                    self.mask = None; self._edges = None
                elif k in (ord('c'), ord('C')):  # clear clicks
                    self.pos.clear(); self.neg.clear()
                elif k in (ord('o'), ord('O')):  # open
                    if self.mask is not None:
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, k3, iterations=1); self._edges = None
                elif k in (ord('k'), ord('K')):  # close
                    if self.mask is not None:
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, k3, iterations=1); self._edges = None
                elif k in (ord('s'), ord('S')):  # save
                    self.save()
                elif k in (ord('n'), ord('N')):  # next
                    self.idx += 1; break
                elif k in (ord('v'), ord('V')):  # prev
                    self.idx -= 1; break
                elif k in (ord('q'), ord('Q')):  # quit
                    cv2.destroyAllWindows(); return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_root', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--sam2_ckpt', required=True)
    ap.add_argument('--sam2_cfg',  required=True)
    ap.add_argument('--device', default=None)

    # --- domain & dirnames ---
    ap.add_argument('--domain', default='failure', choices=['dataset_min_failure/cloth','dataset_failure/cloth','dataset_basket/cloth', 'dataset_wm/cloth'])
    ap.add_argument('--rgb_dirname', default='rgb')
    ap.add_argument('--depth_dirname', default='depth_img')

    # 클릭 → 마스크
    ap.add_argument('--enable_zmask', action='store_true')
    ap.add_argument('--zmax', type=float, default=0.5)

    # 파일명 매핑
    ap.add_argument('--depth_suffix', default='_depth')
    ap.add_argument('--rgb_suffix',   default='_rgb')
    ap.add_argument('--strip_suffix', action='store_true')
    ap.add_argument('--rgb_exts',     default='.png,.jpg,.jpeg')

    # diff 계산 정책
    ap.add_argument('--save_diff_cur', action='store_true')
    ap.add_argument('--diff_mode', default='on_save', choices=['on_save','on_load','off'])
    ap.add_argument('--align', default='affine', choices=['affine','none'])

    # empty 매칭 속도 정책
    ap.add_argument('--ref_match', default='cached', choices=['cached','first','idx','best_ecc'])
    ap.add_argument('--ref_idx', type=int, default=0)
    ap.add_argument('--match_down', type=int, default=4, help='downscale factor for matching (4~8 권장)')
    ap.add_argument('--ref_cache', default=None, help='매칭 결과 캐시 json 경로')

    # empty 위치 직접/자동 설정
    ap.add_argument('--empty_dir', default=None, help='empty 샘플 도메인 루트(안에 depth_dirname 존재)')
    ap.add_argument('--empty_domain', default=None, help='자동 탐색 시 사용할 empty 도메인 폴더명(기본: )')

    args = ap.parse_args()
    App(args)

if __name__ == '__main__':
    main()
