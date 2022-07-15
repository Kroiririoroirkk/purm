import os
import yaml
import numpy as np

import torch

class CameraSystem:
    """Class for calibrated camera system

    Supports perspective projection and optimization-based triangulation.
    
    Args:
        yaml_file: calibration file containing cameras and corresponding intrinsics and extrinsics
        
    Attributes:
        (all attributes are torch.Tensors unless otherwise specified)
        
        num_cams: the number of cameras in the camera system
        R (num_cams, 3, 3): camera rotation, where X_c = RX_w rotates X_w from from world to camera coordinates
        t (num_cams, 3): camera translation, vector from camera origin to world origin in camera coordinates
        focal_length (num_cams): focal length
        camera_center (num_cams, 2): camera center
        distortion (num_cams, 5): distortion coefficients
        K (num_cams, 3, 3): intrinsics matrix
        P (num_cams, 3, 4): projection matrix
    
    """
    def __init__(self, yaml_file=None, device='cpu'):
        
        self.device = device
        
        # cam parameters from yaml
        if yaml_file:
            calibs = yaml.safe_load(open(yaml_file))
        else:
            yaml_file = os.path.join(os.path.dirname(__file__), "aviary_2019-06-01_calibration.yaml")
            calibs = yaml.safe_load(open(yaml_file))
        
        resolution = [np.array(calibs[key]['resolution']) for key in sorted(calibs)]
        self.image_size = resolution[0]
        
        self.cam_names = [calibs[key]['rostopic'] for key in sorted(calibs)]
        self.num_cams = len(self.cam_names)
        
        cam_Ps = torch.tensor([np.array(calibs[key]['T_cam_imu']) for key in sorted(calibs)]).float().to(self.device)
        cam_in = torch.tensor([np.array(calibs[key]['intrinsics']) for key in sorted(calibs)]).float().to(self.device)
        cam_dt = torch.tensor([np.array(calibs[key]['distortion_coeffs']) for key in sorted(calibs)]).float().to(self.device)

        # rotation, translation, focal_length, camera_center
        self.R = cam_Ps[:, :3, :3]
        self.t = cam_Ps[:, :3, 3]
        self.focal = cam_in[:, :2].mean(axis = -1)
        self.center = cam_in[:, 2:]
        self.distortion = cam_dt
        
        self.location = torch.einsum('bij,bi->bj', self.R, -self.t)
        
        self.K = torch.eye(3).repeat(self.num_cams,1,1).float().to(self.device)
        self.K[:,0,0] = self.focal
        self.K[:,1,1] = self.focal
        self.K[:,0:2,2] = self.center
        
        self.P = torch.matmul(self.K, torch.cat((self.R, self.t[:,:,None]),-1))
        
        self.fund_mats = self.calc_fund_mats()
    
    # EPIPOLAR GEOMETRY
    @staticmethod
    def f_from_ps(P1, P2):
        """Get the fundamental matrix between two cameras with projection matrics P1, P2
        
        Args:
            P1 (3, 4) and P2 (3, 4): projection matrices
        
        Returns:
            F (3, 3): the fundamental matrix from P1 to P2
        borrowed from https://www.robots.ox.ac.uk/~vgg/hzbook/code/

        """

        x0 = P1[[1,2],:]
        x1 = P1[[2,0],:]
        x2 = P1[[0,1],:]
        y0 = P2[[1,2],:]
        y1 = P2[[2,0],:]
        y2 = P2[[0,1],:]

        F = torch.tensor([
            [
            torch.det(torch.cat((x0,y0))),
            torch.det(torch.cat((x1,y0))),
            torch.det(torch.cat((x2,y0)))],
            [
            torch.det(torch.cat((x0,y1))),
            torch.det(torch.cat((x1,y1))),
            torch.det(torch.cat((x2,y1)))],
            [
            torch.det(torch.cat((x0,y2))),
            torch.det(torch.cat((x1,y2))),
            torch.det(torch.cat((x2,y2)))]])

        return F
    
    def calc_fund_mats(self):
        """Get a fundamental matrix between all pairs of cameras in the camera system"""
        fund_mats = torch.zeros((self.num_cams, self.num_cams, 3, 3)).float().to(self.device)
        for i, camp1 in enumerate(self.P):
            for j, camp2 in enumerate(self.P):
                fund_mats[i,j,:,:] = self.f_from_ps(camp1, camp2)
        
        return fund_mats
    
    @staticmethod
    def tracklet_epidist(F, p1, p2):
        """Compute epipolar distance between the epipolar line corresponding to p1 and p2 in view 2 using fundamental matrix F
        
        Args:
            F (3,3): fundamental matrix from cam1 (containing p1) to cam2 (containing p2)
            p1 (2): image point whose epipolar p2 will be measured from
            p2 (2): image point from which distance to epipole of p1 will be measured

        """
        line = torch.einsum('ij, kj -> ki',F,p1)
        num = torch.abs(torch.einsum('ki, ki -> k', p2, line))
        denom = torch.norm(line[:,:2], dim = 1, p=2)

        return num / denom
    
    @staticmethod
    def to_homogeneous(tensor):
        try:
            device = tensor.device
            ones_shape = tuple(list(tensor.shape[:-1]) + [1])
            result = torch.cat((tensor, torch.ones(ones_shape).to(device)), axis = -1)
        except:
            tensor = torch.Tensor(tensor)
            ones_shape = tuple(list(tensor.shape[:-1]) + [1])
            result = torch.cat((tensor, torch.ones(ones_shape)), axis = -1)
        return result
    
    # PERSPECTIVE PROJECTION
    def get_visibility(self, points, depth):
        """Test whether points are visible in the image
        
        Args:
            points (n_cams, n_points, 2): [x,y] points to test
            depth (n_cams, n_points, 1): depth of the points
            
        Returns:
            visible_mask (n_cams, n_points, 1): whether the point is visible in each camera

        """
        visible_mask = (depth > 0) * \
                        (points[:,:,0] > 0) * (points[:,:,0] < self.image_size[0]) * \
                        (points[:,:,1] > 0) * (points[:,:,1] < self.image_size[1])

        return visible_mask
    
    def perspective_projection(self, points):
        """Compute the perspective projection of a set of points onto all cameras
        
        Args:
            points (n_points, 3): 3D points
            
        Returns:
            projected_points (n_cams, n_points, 3): image points and whether 
                they are visible [x, y, visible]

        """
        
        if isinstance(points, np.ndarray):
            points = torch.FloatTensor(points)
            return_dtype = "numpy"
        else:
            return_dtype = "tensor"

        if not points.device == self.device:
            return_device = points.device
            points = points.to(self.device)
        else:
            return_device == None
        
        num_points = points.shape[0]
        
        points = points.repeat(self.num_cams,1,1)
        
        #print(f"points_3d:{points[2]}")

        # Extrinsic
        if self.R is not None:
            points = torch.einsum('bij,bkj->bki', self.R, points)
            
        #print(f"points_R:{points[2]}")

        if self.t is not None:
            points = points + self.t.unsqueeze(1)
        
        #print(f"points_t:{points[2]}")

        depth = points[:,:,2].clone()
        
        p2 = points.clone()
        p2 = p2 / points[:,:,2:]
        p2 = torch.einsum('bij,bkj->bki', self.K, p2)
        #print(f"points_proj:{p2[2]}")

        if self.distortion is not None:
            kc = self.distortion
            
            #print(kc[2])
            
            points = points[:,:,:2] / points[:,:,2:]
            #print(f"points_norm:{points[2]}")
            
            # this block isn't needed because it just takes image points and then converts them back to relative coordinates
#             points = torch.einsum('bij,bkj->bki', self.K, points)
#             points[:,:,0] = (points[:,:,0] - self.K[:,0,2,None])/self.K[:,0,0,None]
#             points[:,:,1] = (points[:,:,1] - self.K[:,1,2,None])/self.K[:,1,1,None]
#             print(f"points_b:{points[2]}")

            r2 = points[:,:,0]**2 + points[:,:,1]**2
            dx = (2 * kc[:,[2]] * points[:,:,0] * points[:,:,1] 
                    + kc[:,[3]] * (r2 + 2*points[:,:,0]**2))

            dy = (2 * kc[:,[3]] * points[:,:,0] * points[:,:,1] 
                    + kc[:,[2]] * (r2 + 2*points[:,:,1]**2))

            x = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,0] + dx
            y = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,1] + dy

            points = torch.stack([x, y, torch.ones_like(x)], dim=-1)

        #print(f"points_ud:{points[2]}")
            
        # Apply camera intrinsics
        #points = points / points[:,:,-1].unsqueeze(-1)
        projected_points = torch.einsum('bij,bkj->bki', self.K, points)
        
        undist_vis = self.get_visibility(p2[:,:,:-1], p2[:,:,-1])
        dist_vis = self.get_visibility(projected_points[:,:,:-1], depth)

        projected_points[:,:,-1] = undist_vis * dist_vis
        
        # protect agains bad reprojections
        bad_idx = (torch.isinf(projected_points) + torch.isnan(projected_points)) > 0
        projected_points[bad_idx] = 0

        if return_dtype == "numpy":
            projected_points = projected_points.cpu().numpy()
        elif return_device is not None:
            projected_points = projected_points.to(return_device)

        return projected_points

    # TRIANGULATION
    @staticmethod
    def projection_loss(x, y):
        """Calculate projection loss between 2D reprojected points x and ground truth points y
        
        Args:
            x (n_cams, n_points, 3): [x, y, visible] image coordinates of points
            y (n_cams, n_points, 3): [x, y, visible] ground truth image coordinates
        
        Returns:
            loss (1): total loss

        """
        
        mask = y[:,:,-1,None]
        x_pts = x[:,:,:2].float()
        y_pts = y[:,:,:2].float()
        
        loss = ((x_pts - y_pts) * mask).norm(p=2)
        #loss = torch.mean(torch.sqrt(gmof((x.float() - y.float()) * mask, 100).sum(axis = 1)))
        
        return loss

    @staticmethod
    def get_point_errors(x, y):
        """Calculate mean distance between 2D reprojected points x and detected points y
        
        Args:
            x (n_cams, n_points, 3): [x, y, visible] image coordinates of reprojected points
            y (n_cams, n_points, 3): [x, y, visible] image coordinates of detected points
        
        Returns:
            pt_errors (n_points, 1): mean distance error across all cams for each point in x
        
        """
        mask = y[:,:,-1,None]
        x_pts = x[:,:,:2].float()
        y_pts = y[:,:,:2].float()
        pt_errors = torch.sqrt((torch.square(x_pts - y_pts) * mask).sum(axis = -1).mean(axis = 0))
        
        return pt_errors
    
    def initialize_points(self, points):
        """Triangulate points using svd
        
        Args:
            points (n_cams, n_points, 3): matched points [x, y, visible] across cameras with visible indicating whether the point should be used by the camera.
                If point j is only visible from cams 0 and 3 for instance, then the jth index of the second dimension would be:
                    [[x_0,y_0,1],[0,0,0],[0,0,0],[x_3,y_3,1],...]
        
        Returns:
            X (n_points, 3): triangulated 3d points
        
        """

        points_3d_init = torch.zeros(points.shape[1], 3, dtype = torch.float)
        #points_3d_init = torch.tensor([2.9, 1.4, 1.4]).repeat(1, points.shape[1], 1).to(self.device)

        for point_num in range(points.shape[1]):
            visible_mask = points[:, point_num, 2] == 1
            cam_proj = self.P[visible_mask]
            pp = points[visible_mask, point_num, :2]
            
            if cam_proj.shape[0] < 2:
                continue

            # from: https://github.com/karfly/learnable-triangulation-pytorch
            n_views = len(cam_proj)

            A = cam_proj[:, 2:3].expand(n_views, 2, 4) * pp.view(n_views, 2, 1)
            A -= cam_proj[:, :2]

            u, s, vh = torch.svd(A.view(-1, 4))

            point_3d_homo = -vh[:, 3]
            point_3d_homo = point_3d_homo.unsqueeze(0)
            point_3d = (point_3d_homo.transpose(1, 0)[:-1] / point_3d_homo.transpose(1, 0)[-1]).transpose(1, 0)
            point_3d = point_3d[0]
            
            points_3d_init[point_num] = point_3d

        return points_3d_init

    def triangulate_points(self, points, iters=100):
        """Triangulate points (using masks indicating whether the point is visible in each camera)
        
        Args:
            points (n_cams, n_points, 3): matched points [x, y, visible] across cameras with visible indicating whether the point should be used by the camera.
                If point j is only visible from cams 0 and 3 for instance, then the jth index of the second dimension would be:
                    [[x_0,y_0,1],[0,0,0],[0,0,0],[x_3,y_3,1],...]
        
        Returns:
            X (n_points, 3): triangulated 3d points
            pt_errors (n_points, 1): mean distance error across all cams for each point in X
        
        """

        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points)

        if not points.device == self.device:
            points = points.to(self.device)
            
        n_pts = points.shape[1]

        # initialize 3D points using DLT
        X = self.initialize_points(points)
        points_3d_init = X.detach().clone()
        projected_init = self.perspective_projection(points_3d_init)
        pt_errors_init = self.get_point_errors(projected_init, points)

        X.requires_grad = True

        optimizer = torch.optim.Adam([X], lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
        for i in range(iters):
            # we repeat X along the camera dimension so that it repojects into all cameras
            projected_points = self.perspective_projection(X)
            # loss = projection_loss(projected_points[:,:,:2].squeeze(), points)
            loss = self.projection_loss(projected_points, points)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        X = X.detach().squeeze()
        projected_points = projected_points.detach()
        pt_errors = self.get_point_errors(projected_points, points)
        
        # protect agains bad points
        bad_idx = (torch.isinf(pt_errors) + torch.isnan(pt_errors)) > 0
        pt_errors[bad_idx] = 10000
        X[bad_idx,:] = 0

        # if optimization failed, fall back to using DLT
        init_better_idx = torch.bitwise_and(pt_errors_init < pt_errors, torch.sum(points_3d_init == 0, axis = 1) != 3)
        pt_errors[init_better_idx] = pt_errors_init[init_better_idx]
        X[init_better_idx, :] = points_3d_init[init_better_idx]

        return X, pt_errors