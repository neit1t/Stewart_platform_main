import numpy as np

import main_functions_Stewart_platform as mf


class System_coordinate():

    def __init__(self, coordinates_global):

        self.coordinates_global = np.array(coordinates_global)

        self.transformation_matrix_lower_platform = np.array([])

        self.transformation_matrix_upper_platform = np.array([])

        self.coordinates_lower_platform = np.array([], dtype=np.float16)

        self.coordinates_upper_platform = np.array([], dtype=np.float16)

        self.p = 0

    def coordinate_lower_platform(self, alfa, betta, gamma, x, y, z, R_lower):

        self.transformation_matrix_lower_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for i in np.pi*np.linspace(0, 360, 6, endpoint=False)/180:

            self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                        mf.calculation_new_coordinates(self.transformation_matrix_lower_platform, np.array([R_lower*np.cos(i), R_lower*np.sin(i), 0])))
        self.coordinates_lower_platform = np.append(self.coordinates_lower_platform,
                                                    mf.calculation_new_coordinates(self.transformation_matrix_lower_platform, np.array([0, 0, 0]))).reshape(-1, 3)

    def coordinate_upper_platform(self, alfa, betta, gamma, x, y, z, R_upper):

        self.transformation_matrix_upper_platform = mf.transformation(
            alfa, betta, gamma, x, y, z)

        for i in np.pi*np.linspace(30, 360+30, 6, endpoint=False)/180:

            self.coordinates_upper_platform = np.append(self.coordinates_upper_platform, mf.calculation_new_coordinates(
                self.transformation_matrix_upper_platform, np.array([R_upper*np.cos(i), R_upper*np.sin(i), 0])))

        self.coordinates_upper_platform = np.append(self.coordinates_upper_platform,
                                                    mf.calculation_new_coordinates(self.transformation_matrix_upper_platform, np.array([0, 0, 0]))).reshape(-1, 3)

    def coordinates_in_sistem(self, platform, system):

        Itog = np.array([])

        if platform == 'lower':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for i in self.coordinates_lower_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self.transformation_matrix_lower_platform), i))

                return Itog.reshape(-1, 3)

            else:

                return 'NameError'

        elif platform == 'upper':

            if system == 'global':

                return self.coordinates_lower_platform

            elif system == 'local':

                for i in self.coordinates_upper_platform:

                    Itog = np.append(Itog, mf.calculation_new_coordinates(
                        np.linalg.inv(self.transformation_matrix_lower_platform), i))

                return Itog.reshape(-1, 3)

            else:

                return 'NameError'

    def change_of_position_upper_platform(self, alfa, betta, gamma, x, y, z):

        new_coordinate = mf.calculation_new_coordinates(mf.transformation(
            alfa, betta, gamma, x, y, z), self.coordinates_upper_platform)

        self.coordinates_upper_platform = new_coordinate.reshape(-1, 3)

        return self.coordinates_upper_platform
