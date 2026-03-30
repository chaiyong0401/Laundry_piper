import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("reference/basket.pcd")
points = np.asarray(pcd.points)
print(f"총 {len(points)}개의 포인트가 로드되었습니다.")

# 시각화 + 클릭한 포인트 인덱스 선택 모드
# [Shift + 좌클릭] → [Q]로 종료
o3d.visualization.draw_geometries_with_editing(
    [pcd],
    window_name="Basket Mask Viewer",
    width=400,
    height=200,
    left=100,
    top=100
)

# 선택된 포인트는 draw_geometries_with_editing() 종료 후
# Open3D가 자동으로 'picked_points.txt'에 저장
picked = np.loadtxt("picked_points.txt", dtype=int)
print("[선택된 인덱스와 좌표]")
for idx in picked:
    print(f"index={idx}, coord={points[idx]}")
