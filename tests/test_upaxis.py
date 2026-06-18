"""Validate the up-axis (Z-up -> Y-up) world rotation: it must be a rigid
reorientation — image projections unchanged, points and camera centers rotated
together by the same R_up."""
import numpy as np

# R_up as defined in camera_utils.up_axis_matrix('Y_UP'): (x,y,z)->(x,z,-y)
R_up = np.array([[1.,0,0],[0,0,1],[0,-1,0]])

# proper rotation?
assert abs(np.linalg.det(R_up) - 1.0) < 1e-12
assert np.allclose(R_up @ R_up.T, np.eye(3))
# Blender +Z (up) -> +Y (up); +Y(forward) -> -Z
assert np.allclose(R_up @ [0,0,1], [0,1,0])
assert np.allclose(R_up @ [0,1,0], [0,0,-1])
print("R_up (Y_UP) is a proper rotation mapping Z-up -> +Y: OK")

# NEG_Y_UP (LichtFeld): Blender +Z up -> -Y up, (x,y,z)->(x,-z,y)
R_up_neg = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
assert abs(np.linalg.det(R_up_neg) - 1.0) < 1e-12
assert np.allclose(R_up_neg @ R_up_neg.T, np.eye(3))
assert np.allclose(R_up_neg @ [0,0,1], [0,-1,0])      # ceiling -> -Y
# the two Y-up options are 180° apart (this is the upside-down flip)
assert np.allclose(R_up_neg @ np.array([0,0,1.]), -(R_up @ np.array([0,0,1.])))
print("R_up (NEG_Y_UP) maps Z-up -> -Y, 180 deg from Y_UP: OK")

def rot(rx,ry,rz):
    cx,sx=np.cos(rx),np.sin(rx);cy,sy=np.cos(ry),np.sin(ry);cz,sz=np.cos(rz),np.sin(rz)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]);return Rz@Ry@Rx

fx,fy,cx,cy = 1600.,1600.,960.,540.
rng = np.random.default_rng(3)
maxerr = 0.0
for _ in range(500):
    R = rot(*rng.uniform(-2,2,3))                 # world-to-cam rotation (Blender frame)
    t = rng.uniform(-3,3,3)
    p = rng.uniform(-5,5,3)                        # a world point (Blender frame)
    Xc = R@p + t
    if Xc[2] <= 0.05:   # keep it in front
        continue
    u = fx*Xc[0]/Xc[2]+cx; v = fy*Xc[1]/Xc[2]+cy

    # apply up-axis rotation to BOTH the camera pose and the point
    Rp = R @ R_up.T          # rotate_world_to_cam
    tp = t                   # translation unchanged
    pp = R_up @ p            # point rotated

    Xc2 = Rp@pp + tp
    u2 = fx*Xc2[0]/Xc2[2]+cx; v2 = fy*Xc2[1]/Xc2[2]+cy
    maxerr = max(maxerr, abs(u-u2), abs(v-v2), float(np.linalg.norm(Xc-Xc2)))

    # camera center rotates with the world: C' == R_up @ C
    C  = -R.T @ t
    Cp = -Rp.T @ tp
    assert np.allclose(Cp, R_up @ C, atol=1e-9)

assert maxerr < 1e-9, maxerr
print(f"projection invariant under up-axis rotation (max pixel/coord err {maxerr:.2e}): OK")

# c2w (transforms.json) rotation: new c2w = R_up4 @ c2w ; center = translation column
R_up4 = np.eye(4); R_up4[:3,:3] = R_up
c2w = np.eye(4); c2w[:3,:3] = rot(0.3,-0.5,1.0); c2w[:3,3] = [1.5,-2.0,3.25]
c2w2 = R_up4 @ c2w
assert np.allclose(c2w2[:3,3], R_up @ c2w[:3,3]), "c2w position must rotate by R_up"
assert np.allclose(c2w2[:3,:3] @ c2w2[:3,:3].T, np.eye(3)), "rotation stays orthonormal"
# floor point (high Z in Blender) becomes high Y after conversion
ceiling = np.array([0,0,3.0])
assert (R_up @ ceiling)[1] > 2.9, "ceiling should map to +Y up"
print("c2w rotation + up-direction mapping consistent: OK")

print("\nALL UP-AXIS TESTS PASSED")
