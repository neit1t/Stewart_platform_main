import numpy as np

import main_functions_Stewart_platform as mf

import creating_a_coordinate_system as sc

class Stuart(sc.System_coordinate):

    def __init__(self, coordinates_global, min_len, max_len, min_angle, max_angle):
        super().__init__(coordinates_global)
        self.min_len = min_len
        self.max_len = max_len
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.len = np.array([])
        self.test_lens = np.array([])
        self.test_angle = np.array([])

    def len_leg(self, p=0, lens=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            lens = np.append(lens, mf.len_leg(self.coordinates_lower_platform[p],
                                              i))

            p += 1

        self.len = lens

        self.test_lens = ((self.len > self.min_len) &
                          (self.len < self.max_len))

        return self.test_lens

    def angle(self, p=0, angles=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            angles = np.append(angles, mf.angle_between_vectors(i,
                                                                self.coordinates_lower_platform[p],
                                                                self.coordinates_lower_platform[-1],
                                                                self.coordinates_lower_platform[-1]))

        self.angle_lens = angles.reshape(-1, 2)

        self.test_angle = ((self.angle_lens > self.min_angle) &
                           (self.angle_lens < self.max_angle))

        self.test_angle = self.test_angle.reshape(-1, 2)

        return self.test_angle
