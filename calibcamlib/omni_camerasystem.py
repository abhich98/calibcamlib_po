import numpy as np
from scipy.spatial.transform import Rotation as R
from calibcamlib import OmniCamera
from calibcamlib.helper import intersect, get_line_dist


# R,t are world->cam
class OmniCamerasystem:
    def __init__(self):
        self.cameras = list()

    def add_camera(self, K, xi, D, rotmat, t):
        self.cameras.append({'camera': OmniCamera(K, xi, D), 'R': rotmat, 't': t})

    def project(self, X, offsets=None):
        if offsets is None:
            offsets = np.zeros((len(self.cameras), 2))

        X_shape = X.shape
        X = X.reshape(-1, 3)
        x = np.zeros(shape=(len(self.cameras), X.shape[0], 2))

        for i, (c, o) in enumerate(zip(self.cameras, offsets)):
            coords_cam = (c['R'] @ X.T).T + c['t']
            x[i] = c['camera'].space_to_sensor(coords_cam, o).T.T

        return x.reshape((len(self.cameras),) + X_shape[0:-1] + (2,))

    @staticmethod
    def from_calibcam_file(filename: str):
        cs = OmniCamerasystem()
        calib = np.load(filename, allow_pickle=True)[()]

        for i in range(len(calib['RX1_fit'])):
            K = np.array([
                [calib['K_fit'][i][0], 0, calib['K_fit'][i][1]],
                [0, calib['K_fit'][i][2], calib['K_fit'][i][3]],
                [0, 0, 1]
            ])

            cs.add_camera(K,
                          calib['xi_fit'][i],
                          calib['D_fit'][i],
                          calib['RX1_fit'][i],
                          calib['tX1_fit'][i] * calib['square_size_real']
                          )

        return cs

    @staticmethod
    def from_calibs(calibs):
        cs = OmniCamerasystem()

        for calib in calibs:
            cs.add_camera(calib['A'] if 'A' in calib else calib['K'],
                          calib['xi'],
                          calib['k'] if 'k' in calib else calib['D'],
                          R.from_rotvec(calib['rvec_cam'].reshape((3,))).as_matrix(),
                          calib['tvec_cam'].reshape(1, 3)
                          )

        return cs
