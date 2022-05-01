from data.data import  generate_motion_frames


bucket = "coco"

#download zip files from google bucket and download and process images
generate_motion_frames(bucket, count=20)