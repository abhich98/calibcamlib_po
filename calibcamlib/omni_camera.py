import numpy as np
import cv2


class OmniCamera:
    ID = "opencv-omnidir"
    NAME = "Spherical Camera"

    def __init__(self, K, xi, D, offset=None, distortion=None):  # TODO: Implement variable distortion
        if offset is None:
            offset = [0, 0]

        self.K = K.reshape(3, 3)
        self.xi = xi
        self.D = np.asarray(D).squeeze()[:4]
        self.offset = offset

    def space_to_sensor(self, X, offset=None):
        """Projects points from object space to image space, from camera coordinate system to the image.
        :param X: X.shape = (num_points, 3); even when num_points is 1
        """

        if offset is None:
            offset = self.offset

        # assert self.k[2] == 0 and self.k[3] == 0 and self.k[4] == 0

        x = \
        cv2.omnidir.projectPoints(np.array([X], dtype=np.float32), np.zeros(3), np.zeros(3), self.K, self.xi.ravel()[0], self.D)[0]

        return x.squeeze()

    def sensor_to_space(self, x, offset=None):
        """Reprojects/back-projects points from image space to object space, from the image to normalized camera coordinate system.
        :param x: x.shape = x.shape = (num_points, 2); even when num_points is 1
        """

        if offset == None:
            offset = self.offset

        # assert self.k[2] == 0 and self.k[3] == 0 and self.k[4] == 0

        # TODO: Compare the two funtions.
        #  The omnidir funtion seems to be unstable and the dependance of output on xi is unclear!
        # x_pre = cv2.omnidir.undistortPoints(np.array([x], dtype=np.float32), self.K, self.D, self.xi, np.zeros(3))
        x_pre = cv2.undistortPoints(np.array([x], dtype=np.float32), self.K, self.D)

        X1, X2 = self.pre_to_space(x_pre.squeeze())

        return X1, X2

    def pre_to_space(self, x_pre):

        num_points = x_pre.shape[0]
        xi = np.asarray(self.xi).ravel()[0]

        numtor_1 = np.zeros(num_points)
        numtor_2 = np.zeros(num_points)

        det = ((1 + ((1 - xi ** 2) * (x_pre[:, 0] ** 2 + x_pre[:, 1] ** 2))) >= 0)

        numtor_1[det] = xi + np.sqrt(1 + ((1 - xi ** 2) * (x_pre[:, 0][det] ** 2 + x_pre[:, 1][det] ** 2)))
        numtor_2[det] = xi - np.sqrt(1 + ((1 - xi ** 2) * (x_pre[:, 0][det] ** 2 + x_pre[:, 1][det] ** 2)))

        dentor = x_pre[:, 0] ** 2 + x_pre[:, 1] ** 2 + 1

        object_points_1 = np.zeros((num_points, 3))

        object_points_1[:, 0] = x_pre[:, 0] * (numtor_1 / dentor)
        object_points_1[:, 1] = x_pre[:, 1] * (numtor_1 / dentor)
        object_points_1[:, 2] = (numtor_1 / dentor) - xi

        object_points_2 = np.zeros((num_points, 3))

        object_points_2[:, 0] = x_pre[:, 0] * (numtor_2 / dentor)
        object_points_2[:, 1] = x_pre[:, 1] * (numtor_2 / dentor)
        object_points_2[:, 2] = (numtor_2 / dentor) - xi

        return object_points_1, object_points_2
