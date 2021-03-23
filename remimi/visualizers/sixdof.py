import numpy as np
import open3d as o3d


class OnahoPointCloudVisualizer:
    def __init__(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis_pcd = o3d.geometry.PointCloud()
        vis_bounding_box = o3d.geometry.OrientedBoundingBox()
        vis_bounding_box.extent = np.array([1,1,1])
        self.lineset = o3d.geometry.LineSet()
        vis.add_geometry(vis_pcd)
        vis.add_geometry(vis_bounding_box)
        vis.add_geometry(self.lineset)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
        # vis.add_geometry(coord)


        self.vis = vis
        self.vis_bounding_box = vis_bounding_box
        self.vis_pcd = vis_pcd

    def change_bounding_box_color(self, color):
        self.vis_bounding_box.color = color

    def update_bounding_box(self, closest_bounding_box: o3d.geometry.OrientedBoundingBox):
        vis_bounding_box = self.vis_bounding_box
        vis_bounding_box.extent = closest_bounding_box.extent
        vis_bounding_box.center = closest_bounding_box.center
        vis_bounding_box.color = closest_bounding_box.color
        vis_bounding_box.R = closest_bounding_box.R
        self.vis.update_geometry(vis_bounding_box)

    def update_axis(self, points):
        self.lineset.points = o3d.utility.Vector3dVector(np.array(points))
        self.lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        self.vis.update_geometry(self.lineset)

    def update_pcd(self, pcd: o3d.geometry.PointCloud):
        vis, vis_pcd = self.vis, self.vis_pcd

        vis_pcd.points = pcd.points
        vis_pcd.colors = pcd.colors
        vis.update_geometry(vis_pcd)
        vis.poll_events()
        vis.update_renderer()
