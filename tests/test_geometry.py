"""Validate depth back-projection round-trip and barycentric sampling."""
import numpy as np

R_BCAM2CV = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)

def rot(rx,ry,rz):
    cx,sx=np.cos(rx),np.sin(rx);cy,sy=np.cos(ry),np.sin(ry);cz,sz=np.cos(rz),np.sin(rz)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]);return Rz@Ry@Rx

# Build a camera-to-world, derive world-to-cam OpenCV (R,t) as the addon does.
Rcw=rot(0.4,-0.6,1.0); loc=np.array([2.0,-1.0,1.6])
M=np.eye(4);M[:3,:3]=Rcw;M[:3,3]=loc
w2b=np.linalg.inv(M)
R=R_BCAM2CV@w2b[:3,:3]; t=R_BCAM2CV@w2b[:3,3]
fx,fy,cx,cy=1662.77,1662.77,960.0,540.0

# Pick world points in front of the camera, project to pixels+depth, unproject.
rng=np.random.default_rng(0)
errs=[]
for _ in range(2000):
    # random point in camera frame with positive depth, within FOV
    z=rng.uniform(0.5,8.0)
    u=rng.uniform(0,1920); v=rng.uniform(0,1080)
    xc=(u-cx)/fx*z; yc=(v-cy)/fy*z
    Pc=np.array([xc,yc,z])
    Pw=R.T@(Pc-t)                      # true world point
    # ---- forward project (what Blender's depth+render would give) ----
    Pc2=R@Pw+t
    u2=fx*Pc2[0]/Pc2[2]+cx; v2=fy*Pc2[1]/Pc2[2]+cy; depth=Pc2[2]
    # ---- addon unprojection (row form, batched) ----
    xc2=(u2-cx)/fx*depth; yc2=(v2-cy)/fy*depth
    pc=np.array([[xc2,yc2,depth]])
    pw=(pc - t[None,:]) @ R            # <-- exact formula from pointcloud.py
    errs.append(np.linalg.norm(pw[0]-Pw))
print(f"Depth unprojection round-trip max err: {max(errs):.2e}  mean: {np.mean(errs):.2e}")
assert max(errs)<1e-9, "unprojection mismatch"
print("Depth back-projection (pc - t) @ R: OK")

# --- barycentric sampling: points stay inside the triangle ---
v0=np.array([0.,0,0]);v1=np.array([1.,0,0]);v2=np.array([0.,1,0])
n=10000;u1=rng.random(n);u2=rng.random(n);su1=np.sqrt(u1)
b0=(1-su1);b1=su1*(1-u2);b2=su1*u2
assert np.allclose(b0+b1+b2,1.0)
assert (b0>=-1e-12).all() and (b1>=-1e-12).all() and (b2>=-1e-12).all()
pts=b0[:,None]*v0+b1[:,None]*v1+b2[:,None]*v2
# all points inside unit right triangle: x>=0,y>=0,x+y<=1
assert (pts[:,0]>=-1e-12).all() and (pts[:,1]>=-1e-12).all() and (pts[:,0]+pts[:,1]<=1+1e-9).all()
print("Barycentric sampling stays inside triangle: OK")

# --- area-weighted distribution sanity: big triangle gets ~proportional samples ---
areas=np.array([1.0,3.0])   # two triangles, 1:3 area
p=areas/areas.sum()
idx=rng.choice(2,size=400000,p=p)
frac=(idx==1).mean()
assert abs(frac-0.75)<0.01, frac
print(f"Area-weighted choice ~proportional (got {frac:.3f}, want 0.75): OK")

# --- lin->srgb monotonic & endpoints ---
def lin2srgb(c):
    c=np.clip(c,0,1);return np.where(c<=0.0031308,c*12.92,1.055*np.power(c,1/2.4)-0.055)
assert abs(lin2srgb(np.array([0.0]))[0]-0)<1e-9
assert abs(lin2srgb(np.array([1.0]))[0]-1)<1e-9
assert abs(lin2srgb(np.array([0.5]))[0]-0.7353569)<1e-4
print("linear->sRGB transform: OK")

print("\nALL GEOMETRY TESTS PASSED")
