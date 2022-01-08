import numpy as np

import main_functions_Stewart_platform as mf

import Errors as er


class Stuart():

    def __init__(self, coordinates_global, min_len, max_len, angles):
        self.min_len = er.Check._Check__checkNumbers(min_len)
        self.max_len = er.Check._Check__checkNumbers(max_len)
        self.angles = er.Check._Check__checkNumbers(angles)
        self.len = np.array([])
        self.test_lens = np.array([])
        self.test_angle = np.array([])
        self.coordinates_global = er.Check._Check__checkShape(
            er.Check._Check__checkArrays(coordinates_global))
        self._Stuart__transformation_matrix_lower_platform = np.array([])
        self._Stuart__transformation_matrix_upper_platform = np  .array([])

        self.coordinates_lower_platform = np.array([], dtype=np.float16)

        self.coordinates_upper_platform = np.array([], dtype=np.float16)

        self.angls_and_moovs_lower_platform = np.array([])

        self.angls_and_moovs_upper_platform = np.array([])

        print('Вы задали', f'Координаты глобальной системы:{coordinates_global}',
         f'Минимальная и максимальная длинна ноги:{min_len,max_len}',
              f'Минимальный и максимальный наклон ног:{(90-self.angles),(90+self.angles)}',
               sep='\n')

    def coordinate_lower_platform(self, R_lower, alfa, betta, gamma, x, y, z):

        list(map(er.Check._Check__checkNumbers, [
             alfa, betta, gamma, x, y, z, R_lower]))
        self._Stuart__transformation_matrix_lower_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for angle_lower in np.pi*np.linspace(0, 360, 6, endpoint=False)/180:

            self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                        mf.calculation_new_coordinates(self._Stuart__transformation_matrix_lower_platform, np.array([R_lower*np.cos(angle_lower), R_lower*np.sin(angle_lower), 0])))
        self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                    mf.calculation_new_coordinates(self._Stuart__transformation_matrix_lower_platform, np.array([0, 0, 0]))).reshape(-1, 3)
        self.angls_and_moovs_lower_platform = np.append(
            self.angls_and_moovs_lower_platform, [alfa, betta, gamma, x, y, z])

    def coordinate_upper_platform(self, R_upper, alfa, betta, gamma, x, y, z):

        list(map(er.Check._Check__checkNumbers, [
             alfa, betta, gamma, x, y, z, R_upper]))

        self._Stuart__transformation_matrix_upper_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for angle_upper in np.pi*np.linspace(30, 360+30, 6, endpoint=False)/180:

            self.coordinates_upper_platform = np.append(self.coordinates_upper_platform, mf.calculation_new_coordinates(
                self._Stuart__transformation_matrix_upper_platform, np.array([R_upper*np.cos(angle_upper), R_upper*np.sin(angle_upper), 0])))

        self.coordinates_upper_platform = np.append(self.coordinates_upper_platform,
                                                    mf.calculation_new_coordinates(self._Stuart__transformation_matrix_upper_platform, np.array([0, 0, 0]))).reshape(-1, 3)
        self.angls_and_moovs_upper_platform = np.append(
            self.angls_and_moovs_upper_platform, [alfa, betta, gamma, x, y, z])

    def coordinates_in_sistem(self, platform, system):

        er.Check._Check__checkNames(platform, system)

        Itog = np.array([])

        if platform == 'lower':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for coordinate_lower_leg in self.coordinates_lower_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self._Stuart__transformation_matrix_lower_platform), coordinate_lower_leg))

                return Itog.reshape(-1, 3)

        elif platform == 'upper':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for coordinate_upper_leg in self.coordinates_upper_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self._Stuart__transformation_matrix_lower_platform), coordinate_upper_leg))

                return Itog.reshape(-1, 3)

    def change_of_position_upper_platform(self, alfa, betta, gamma, x, y, z):

        list(map(er.Check._Check__checkNumbers, [alfa, betta, gamma, x, y, z]))

        self.__transform_matrix = mf.transformation(alfa, betta, gamma, x, y, z)

        self.__inv_matrix = np.linalg.inv(self.__transform_matrix)

        new_coordinate = mf.calculation_new_coordinates(self._Stuart__transform_matrix, self.coordinates_upper_platform)

        self.coordinates_upper_platform = new_coordinate.reshape(-1, 3)

        self.angls_and_moovs_upper_platform = np.append(
            self.angls_and_moovs_upper_platform, [alfa, betta, gamma, x, y, z])

        return self.coordinates_upper_platform

    def len_leg(self, test=False, _Stuart__p=0, __lens=np.array([])):

        for coordinate_upper_leg in self.coordinates_upper_platform[:-1]:

            _Stuart__lens = np.append(_Stuart__lens, mf.len_leg(self.coordinates_lower_platform[_Stuart__p],
                                                                coordinate_upper_leg))

        self.len = _Stuart__lens

        self.test_lens = ((self.len > self.min_len) &
                          (self.len < self.max_len))

        if test == True:

            print(self.test_lens)

            return self.len

        elif test == False:

            return self.len

        else:

            raise KeyError('Только булевое значение')

    def angle(self, test=False, _Stuart__p=0, _Stuart__angles=np.array([])):

        for coordinate_upper_leg in self.coordinates_upper_platform[:-1]:

            _Stuart__angles = np.append(_Stuart__angles, mf.angle_between_vectors(coordinate_upper_leg,
                                                                                  self.coordinates_lower_platform[_Stuart__p],
                                                                                  self.coordinates_lower_platform[-1],
                                                                                  self.coordinates_lower_platform[-1]))

            _Stuart__p = _Stuart__p + 1

        self.angle_lens = _Stuart__angles.reshape(-1, 2)

        self.test_angle = ((self.angle_lens > (90-self.angles)) &
                           (self.angle_lens < (90+self.angles)))

        self.test_angle = self.test_angle.reshape(-1, 2)

        if test == True:

            print(self.test_angle)

            return self.angle_lens

        elif test == False:

            return self.angle_lens

        else:

            raise KeyError('Только булевое значение')
