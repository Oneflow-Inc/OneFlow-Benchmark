import os

MX_BLOBS_PATH = "/home/scxfjiang/Desktop/mx_blobs"

os.system("rm -rf /home/scxfjiang/Desktop/mx_blobs")

if not os.path.exists(MX_BLOBS_PATH):
    os.mkdir(MX_BLOBS_PATH)

os.system("scp vs004:/home/xfjiang/log/mx_images.npy /home/scxfjiang/Desktop/mx_blobs/")
os.system("scp vs004:/home/xfjiang/log/mx_labels.npy /home/scxfjiang/Desktop/mx_blobs/")
os.system("scp vs004:/home/xfjiang/log/conv0_out.npy /home/scxfjiang/Desktop/mx_blobs/")
os.system("scp vs004:/home/xfjiang/log/fc1_out.npy /home/scxfjiang/Desktop/mx_blobs/")
os.system("scp -r vs004:/home/xfjiang/log/grad /home/scxfjiang/Desktop/mx_blobs/")
