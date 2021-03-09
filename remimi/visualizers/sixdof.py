import numpy as np
import open3d as o3d


class OnahoPointCloudVisualizer:
    def __init__(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis_pcd = o3d.geometry.PointCloud()
        vis_bounding_box = o3d.geometry.OrientedBoundingBox()
        vis_bounding_box.extent = np.array([1,1,1])
        vis.add_geometry(vis_pcd)
        vis.add_geometry(vis_bounding_box)

        self.vis = vis
        self.vis_bounding_box = vis_bounding_box
        self.vis_pcd = vis_pcd

    def update_bounding_box(self, closest_bounding_box: o3d.geometry.OrientedBoundingBox):
        vis_bounding_box = self.vis_bounding_box
        vis_bounding_box.extent = closest_bounding_box.extent
        vis_bounding_box.center = closest_bounding_box.center
        vis_bounding_box.R = closest_bounding_box.R
        self.vis.update_geometry(vis_bounding_box)

    def update_pcd(self, pcd: o3d.geometry.PointCloud):
        vis, vis_pcd = self.vis, self.vis_pcd

        vis_pcd.points = pcd.points
        vis_pcd.colors = pcd.colors
        vis.update_geometry(vis_pcd)
        vis.poll_events()
        vis.update_renderer()
