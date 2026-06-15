"""Verify the Blender->COLMAP camera math (mirrors camera_utils.py with numpy).
Checks the invariants that actually matter for a correct COLMAP export."""
import numpy as np

R_BCAM2CV = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)

def mat_to_quat(R):
    # Standard matrix->quaternion (w,x,y,z), same convention as mathutils.
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z]); return q/np.linalg.norm(q)

def quat_to_mat(q):
    w,x,y,z=q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def rot_euler_xyz(rx,ry,rz):
    cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry); cz,sz=np.cos(rz),np.sin(rz)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx  # Blender default XYZ euler order

def extrinsics(matrix_world):
    w2b = np.linalg.inv(matrix_world)
    R_w2bc = w2b[:3,:3]; t_w2bc = w2b[:3,3]
    R = R_BCAM2CV @ R_w2bc
    t = R_BCAM2CV @ t_w2bc
    return R, t

# --- Test 1: camera at (1.5,-2,3.25), looking straight down -Z, no rotation ---
loc = np.array([1.5,-2.0,3.25])
Rcw = rot_euler_xyz(0,0,0)            # camera-to-world rotation
M = np.eye(4); M[:3,:3]=Rcw; M[:3,3]=loc
R,t = extrinsics(M)
C = -R.T @ t                          # recovered camera center
assert np.allclose(C, loc), f"center mismatch {C} vs {loc}"
print("Test1 camera-center round-trip: OK   center =", np.round(C,4))

# A world point 5m in front of camera. Blender cam looks down -Z (world here),
# so 'in front' = loc + (0,0,-5). It must have POSITIVE depth in OpenCV space.
Pw = loc + np.array([0,0,-5.0])
Pc = R @ Pw + t
assert Pc[2] > 0, f"point in front should have +Z depth, got {Pc}"
assert abs(Pc[0])<1e-9 and abs(Pc[1])<1e-9, f"on-axis point should be centered, got {Pc}"
assert abs(Pc[2]-5.0)<1e-9, f"depth should be 5.0, got {Pc[2]}"
print("Test1 projection (in-front => +Z depth): OK   Pc =", np.round(Pc,4))

# Blender 'up' (+Y world, since no rotation cam up is +Y) maps to OpenCV -Y (down).
Pup = loc + np.array([0,1.0,-5.0])    # 5 ahead, 1 up
Pc_up = R @ Pup + t
assert Pc_up[1] < 0, f"world-up should map to image-up (negative Y), got {Pc_up}"
print("Test1 up-vector (world +Y => OpenCV -Y): OK   Pc_up =", np.round(Pc_up,4))

# --- Test 2: arbitrary rotated camera, center still round-trips ---
for rx,ry,rz in [(0.3,-0.7,1.1),(1.2,0.4,-0.9),(np.pi/2,0,0)]:
    Rcw = rot_euler_xyz(rx,ry,rz)
    M = np.eye(4); M[:3,:3]=Rcw; M[:3,3]=np.array([3.0,4.0,-1.0])
    R,t = extrinsics(M)
    C = -R.T @ t
    assert np.allclose(C,[3,4,-1]), f"center mismatch for {(rx,ry,rz)}: {C}"
    # R must be a proper rotation (det=+1, orthonormal)
    assert abs(np.linalg.det(R)-1.0)<1e-9, f"det(R)={np.linalg.det(R)}"
    assert np.allclose(R@R.T, np.eye(3)), "R not orthonormal"
    # quaternion round-trip
    q = mat_to_quat(R); Rq = quat_to_mat(q)
    assert np.allclose(Rq, R, atol=1e-9), "quat round-trip failed"
print("Test2 rotated cameras (center + orthonormal R + quat round-trip): OK")

# --- Test 3: intrinsics for 50mm lens, 36mm sensor, 1920x1080 HORIZONTAL ---
def intrinsics(f_mm, sensor_w, sensor_h, fit, W, H, pa=1.0):
    if fit=='AUTO': fit='HORIZONTAL' if W>=H*pa else 'VERTICAL'
    if fit=='HORIZONTAL':
        fx=f_mm*W/sensor_w; fy=fx*pa
    else:
        fy=f_mm*H/sensor_h; fx=fy/pa
    return fx,fy,W/2,H/2
fx,fy,cx,cy = intrinsics(50,36,24,'HORIZONTAL',1920,1080)
assert abs(fx-2666.6667)<1e-3 and abs(fy-fx)<1e-9, (fx,fy)
assert (cx,cy)==(960,540)
# horizontal FOV check: 2*atan(36/2/50) ~ 39.6 deg ; fx=W/(2 tan(fov/2))
import math
fov = 2*math.atan((36/2)/50)
assert abs(fx - (1920/2)/math.tan(fov/2)) < 1e-6
print(f"Test3 intrinsics 50mm/36mm/1920x1080: fx=fy={fx:.4f} cx,cy={cx},{cy}: OK")

# AUTO with portrait resolution should pick VERTICAL using sensor_height
fx2,fy2,_,_ = intrinsics(50,36,24,'AUTO',1080,1920)
assert abs(fy2-50*1920/24)<1e-6 and abs(fx2-fy2)<1e-9
print(f"Test3 AUTO portrait -> VERTICAL fit: fx=fy={fx2:.4f}: OK")

print("\nALL CAMERA-MATH TESTS PASSED")
