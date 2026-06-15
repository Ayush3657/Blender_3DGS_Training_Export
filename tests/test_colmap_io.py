"""Round-trip test: write COLMAP bin/txt with our writers, read back with a
reference reader (ported from COLMAP's read_write_model.py), assert equality."""
import os, sys, struct, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "blender_3dgs_export"))
import colmap_io

# ---- reference readers (COLMAP canonical) ----
def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)

def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            cid, model_id, w, h = read_next_bytes(f, 24, "iiQQ")
            # PINHOLE = 4 params
            num_params = 4
            params = read_next_bytes(f, 8*num_params, "d"*num_params)
            cams[cid] = (model_id, w, h, params)
    return cams

def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            iid = read_next_bytes(f, 4, "i")[0]
            qvec = read_next_bytes(f, 32, "dddd")
            tvec = read_next_bytes(f, 24, "ddd")
            cam_id = read_next_bytes(f, 4, "i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c; c = f.read(1)
            num2d = read_next_bytes(f, 8, "Q")[0]
            f.read(24*num2d)  # skip x,y,id triples
            imgs[iid] = (qvec, tvec, cam_id, name.decode())
    return imgs

def read_points3D_bin(path):
    pts = {}
    with open(path, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            pid = read_next_bytes(f, 8, "Q")[0]
            xyz = read_next_bytes(f, 24, "ddd")
            rgb = read_next_bytes(f, 3, "BBB")
            err = read_next_bytes(f, 8, "d")[0]
            tl = read_next_bytes(f, 8, "Q")[0]
            f.read(8*tl)
            pts[pid] = (xyz, rgb, err)
    return pts

# ---- sample data ----
cameras = [
    {'id':1,'model':'PINHOLE','width':1920,'height':1080,'params':[1662.768,1662.768,960.0,540.0]},
    {'id':2,'model':'PINHOLE','width':800,'height':600,'params':[700.5,700.5,400.0,300.0]},
]
images = [
    {'id':1,'qvec':(0.7071,0.0,0.7071,0.0),'tvec':(1.5,-2.0,3.25),'camera_id':1,'name':'GSCam_000.png'},
    {'id':2,'qvec':(1.0,0.0,0.0,0.0),'tvec':(0.0,0.0,0.0),'camera_id':2,'name':'GSCam_001.png'},
]
xyz = np.array([[1.0,2.0,3.0],[-4.5,5.5,-6.5],[0.1,0.2,0.3]], dtype=np.float64)
rgb = np.array([[10,20,30],[200,150,100],[255,0,128]], dtype=np.uint8)

d = tempfile.mkdtemp()
colmap_io.write_model(d, cameras, images, xyz, rgb, fmt='BOTH')
print("Files written:", sorted(os.listdir(d)))

# ---- verify BIN ----
rc = read_cameras_bin(os.path.join(d,"cameras.bin"))
assert rc[1][0]==1 and rc[1][1]==1920 and rc[1][2]==1080
assert abs(rc[1][3][0]-1662.768)<1e-6
assert rc[2][1]==800
ri = read_images_bin(os.path.join(d,"images.bin"))
assert ri[1][3]=='GSCam_000.png'
assert abs(ri[1][0][2]-0.7071)<1e-6
assert abs(ri[1][1][2]-3.25)<1e-9
assert ri[1][2]==1
rp = read_points3D_bin(os.path.join(d,"points3D.bin"))
assert len(rp)==3
assert rp[2][0]==(-4.5,5.5,-6.5), rp[2][0]
assert rp[3][1]==(255,0,128), rp[3][1]
print("BIN round-trip: OK")

# ---- verify TXT parse ----
def parse_cameras_txt(path):
    out={}
    for line in open(path):
        if line.startswith('#') or not line.strip(): continue
        t=line.split(); out[int(t[0])]=(t[1],int(t[2]),int(t[3]),[float(x) for x in t[4:]])
    return out
ct = parse_cameras_txt(os.path.join(d,"cameras.txt"))
assert ct[1][0]=='PINHOLE' and ct[1][1]==1920 and abs(ct[1][3][0]-1662.768)<1e-6

img_lines=[l for l in open(os.path.join(d,"images.txt")) if not l.startswith('#')]
# pairs of (data line, empty line)
data = img_lines[0].split()
assert data[9]=='GSCam_000.png' and int(data[8])==1
pt_lines=[l for l in open(os.path.join(d,"points3D.txt")) if not l.startswith('#') and l.strip()]
assert len(pt_lines)==3
last=pt_lines[1].split()
assert int(last[4])==200 and int(last[5])==150 and int(last[6])==100
print("TXT round-trip: OK")

# ---- PLY ----
colmap_io.write_points_ply(os.path.join(d,"p.ply"), xyz, rgb)
hdr=open(os.path.join(d,"p.ply"),'rb').read(200).decode('ascii',errors='ignore')
assert 'element vertex 3' in hdr
print("PLY header: OK")
print("\nALL TESTS PASSED")
