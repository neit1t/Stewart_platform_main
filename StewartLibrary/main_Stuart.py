import numpy as np

import main_functions_Stewart_platform as mf

import Errors as er


class Stuart():

    def __init__(self, coordinates_global, min_len, max_len, min_angle, max_angle):
        self.min_len = er.Check._Check__checkNumbers(min_len)
        self.max_len = er.Check._Check__checkNumbers(max_len)
        self.min_angle = er.Check._Check__checkNumbers(min_angle)
        self.max_angle = er.Check._Check__checkNumbers(max_angle)
        self.len = np.array([])
        self.test_lens = np.array([])
        self.test_angle = np.array([])
        self.coordinates_global = er.Check._Check__checkShape(
            er.Check._Check__checkArrays(coordinates_global))
        self.transformation_matrix_lower_platform = np.array([])
        self.transformation_matrix_upper_platform = np  .array([])

        self.coordinates_lower_platform = np.array([], dtype=np.float16)

        self.coordinates_upper_platform = np.array([], dtype=np.float16)

        self.angls_and_moovs_lower_platform = np.array([])

        self.angls_and_moovs_upper_platform = np.array([])

    def coordinate_lower_platform(self, alfa, betta, gamma, x, y, z, R_lower):

        list(map(er.Check._Check__checkNumbers, [
             alfa, betta, gamma, x, y, z, R_lower]))
        self.transformation_matrix_lower_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for i in np.pi*np.linspace(0, 360, 6, endpoint=False)/180:

            self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                        mf.calculation_new_coordinates(self.transformation_matrix_lower_platform, np.array([R_lower*np.cos(i), R_lower*np.sin(i), 0])))
        self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                    mf.calculation_new_coordinates(self.transformation_matrix_lower_platform, np.array([0, 0, 0]))).reshape(-1, 3)
        self.angls_and_moovs_lower_platform = np.append(
            self.angls_and_moovs_lower_platform, [alfa, betta, gamma, x, y, z])

    def coordinate_upper_platform(self, alfa, betta, gamma, x, y, z, R_upper):

        list(map(er.Check._Check__checkNumbers, [
             alfa, betta, gamma, x, y, z, R_upper]))

        self.transformation_matrix_upper_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for i in np.pi*np.linspace(30, 360+30, 6, endpoint=False)/180:

            self.coordinates_upper_platform = np.append(self.coordinates_upper_platform, mf.calculation_new_coordinates(
                self.transformation_matrix_upper_platform, np.array([R_upper*np.cos(i), R_upper*np.sin(i), 0])))

        self.coordinates_upper_platform = np.append(self.coordinates_upper_platform,
                                                    mf.calculation_new_coordinates(self.transformation_matrix_upper_platform, np.array([0, 0, 0]))).reshape(-1, 3)
        self.angls_and_moovs_upper_platform = np.append(
            self.angls_and_moovs_upper_platform, [alfa, betta, gamma, x, y, z])

    def coordinates_in_sistem(self, platform, system):

        er.Check._Check__checkNames(platform, system)

        Itog = np.array([])

        if platform == 'lower':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for i in self.coordinates_lower_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self.transformation_matrix_lower_platform), i))

                return Itog.reshape(-1, 3)

        elif platform == 'upper':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for i in self.coordinates_upper_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self.transformation_matrix_lower_platform), i))

                return Itog.reshape(-1, 3)

    def change_of_position_upper_platform(self, alfa, betta, gamma, x, y, z):

        list(map(er.Check._Check__checkNumbers, [alfa, betta, gamma, x, y, z]))

        new_coordinate = mf.calculation_new_coordinates(mf.transformation(
            alfa, betta, gamma, x, y, z), self.coordinates_upper_platform)

        self.coordinates_upper_platform = new_coordinate.reshape(-1, 3)

        self.angls_and_moovs_upper_platform = np.append(
            self.angls_and_moovs_upper_platform, [alfa, betta, gamma, x, y, z])

        return self.coordinates_upper_platform

    def len_leg(self,test = False, _Stuart__p=0, __lens=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            _Stuart__lens = np.append(_Stuart__lens, mf.len_leg(self.coordinates_lower_platform[_Stuart__p],
                                                                i))

        self.len = _Stuart__lens

        self.test_lens = ((self.len > self.min_len) &
                          (self.len < self.max_len))

        if test == True:

            print(self.test_lens)

            return self.len

        elif test == False:

            return self.len

        else:

            raise ValueError('Только булевое значение')

    def angle(self,test = False, _Stuart__p=0, _Stuart__angles=np.array([])):

        for i in self.coordinates_upper_platform[:-1]:

            _Stuart__angles = np.append(_Stuart__angles, mf.angle_between_vectors(i,
                                                                                  self.coordinates_lower_platform[_Stuart__p],
                                                                                  self.coordinates_lower_platform[-1],
                                                                                  self.coordinates_lower_platform[-1]))

        self.angle_lens = _Stuart__angles.reshape(-1, 2)

        self.test_angle = ((self.angle_lens > self.min_angle) &
                           (self.angle_lens < self.max_angle))

        self.test_angle = self.test_angle.reshape(-1, 2)

        if test == True:

            print(self.test_angle)

            return self.angle_lens 

        elif test == False:

            return self.angle_lens

        else:

            raise ValueError('Только булевое значение')