import open3d as o3d

def create_point_cloud_from_color_and_depth(color, depth, intrinsic):
    depth_image = o3d.geometry.Image(depth)
    color_image = o3d.geometry.Image(color)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,                
        depth_scale=1000,
        convert_rgb_to_intensity=False)
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    return temp