
from remimi.edit.hifill.hifill import MaskEliminator
import cv2
import numpy as np
import open3d as o3d

class SimplePointCloudVisualizer:
    def __init__(self, show_axis=False, original_coordinate=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)
        self.vis = vis
        self.original_coordinate = original_coordinate
        
        if show_axis:
            self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
            vis.add_geometry(self.coord)
        self.pcd = o3d.geometry.PointCloud()

        self.frame_count = 0
        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        self.stop = False

    def update_by_pcd(self, temp):
        if not self.original_coordinate:
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

def create_mask(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask[gray != 255] = 255

   #@  import IPython; IPython.embed()

    return mask

class InpaintedPointCloudVisualizer(SimplePointCloudVisualizer):
    def __init__(self):
        super().__init__()
        self.eliminator = MaskEliminator()

    def update_by_pcd(self, pcd):
        super().update_by_pcd(pcd)

        image = np.array(self.vis.capture_screen_float_buffer(False)) * 255
        aa = create_mask(image).astype(np.uint8)
        # kernel = np.ones((5,5),np.uint8)
        # aa = cv2.erode(aa,kernel,iterations = 2)
        # # import IPython; IPython.embed()
        mask = cv2.cvtColor(aa, cv2.COLOR_GRAY2BGR)
        inpainted_image = self.eliminator.eliminate_by_mask(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask)
        cv2.imshow("inpainted", inpainted_image)