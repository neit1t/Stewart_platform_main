import numpy as np

import main_functions_Stewart_platform as mf

import creating_a_coordinate_system as sc

import Errors as er

class Stuart(sc.System_coordinate):

    def __init__(self, coordinates_global, min_len, max_len, min_angle, max_angle):
        super().__init__(coordinates_global)
        self.min_len = er.Check._Check__checkNumbers(min_len)
        self.max_len = er.Check._Check__checkNumbers(max_len)
        self.min_angle = er.Check._Check__checkNumbers(min_angle)
        self.max_angle = er.Check._Check__checkNumbers(max_angle)
        self.len = np.array([])
        self.test_lens = np.array([])
        self.test_angle = np.array([])

    def len_leg(self, __p=0, __lens=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            _Stuart__lens = np.append(_Stuart__lens, mf.len_leg(self.coordinates_lower_platform[_Stuart__p],
                                              i))

            _Stuart__p += 1

        self.len = _Stuart__lens

        self.test_lens = ((self.len > self.min_len) &
                          (self.len < self.max_len))

        return self.test_lens

    def angle(self, _Stuart__p=0, _Stuart__angles=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            _Stuart__angles = np.append(_Stuart__angles, mf.angle_between_vectors(i,
                                                                self.coordinates_lower_platform[_Stuart__p],
                                                                self.coordinates_lower_platform[-1],
                                                                self.coordinates_lower_platform[-1]))

        self.angle_lens = _Stuart__angles.reshape(-1, 2)

        self.test_angle = ((self.angle_lens > self.min_angle) &
                           (self.angle_lens < self.max_angle))

        self.test_angle = self.test_angle.reshape(-1, 2)

        return self.test_angle