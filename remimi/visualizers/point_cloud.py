import open3d as o3d

class SimplePointCloudVisualizer:
    def __init__(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        self.vis = vis

        self.pcd = o3d.geometry.PointCloud()

        self.frame_count = 0
        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        self.stop = False

    def update_by_pcd(self, temp):
        temp.transform(self.flip_transform)
        self.pcd.points = temp.points
        self.pcd.colors = temp.colors
        if self.frame_count == 0:
            self.vis.add_geometry(self.pcd)

        if not self.stop:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.frame_count += 1

    def stop_update(self):
        self.stop = True