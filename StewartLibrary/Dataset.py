import main_Stuart as ms

import main_functions_Stewart_platform as mf

import numpy as np

import pandas as pd


# Класс для выведения датасета как переменной
class Model_Stuart():

    def __init__(self, model, n):
        # Создание модели стюарта
        self.model = model
        # Сохранение результатов
        self.results = np.array([])
        # Сохранение промежуточных результатов
        self.intermediate_result = np.array([])
        # Создание цикла для формирования датасета

        self.n = n

        self.x = np.array([])

        self.y = np.array([])

    def dataset(self, pandas_df=True, prints=False):
        while self.n != 0:
            # Генерация перемещений
            coordinates = 0.1 * \
                ((np.random.rand(3)*self.model.max_len)-self.model.max_len/2)
            # Генерация углов поворота
            angle = 90*np.random.rand(3) - 45
            # Сохранение координат верхней платформы
            self.model.change_of_position_upper_platform(alfa=angle[0], betta=angle[1], gamma=angle[2],

                                                         x=coordinates[0], y=coordinates[1], z=coordinates[2])
            # Расчитываем углы
            self.model.angle()
            # Расчитываем длинны
            self.model.len_leg()

            if sum(self.model.test_lens) == 6 and sum(sum(self.model.test_angle)) == 12:

                self.intermediate_result = np.append((np.concatenate((np.concatenate(
                    (angle, coordinates)), self.model.len))), self.model.angle_lens.reshape(1, -1))

                self.intermediate_result = self.intermediate_result.reshape(
                    1, -1)

                self.results = np.append(
                    self.results, self.intermediate_result)

                self.n -= 1

            else:
                if prints == True:
                    print('Длинны ног', self.model.len, self.model.test_lens, 'Углы наклона ног',
                          self.model.angle_lens, self.model.test_angle, '--------------------', sep='\n')
                self.model.coordinates_upper_platform = mf.calculation_new_coordinates(
                    self.model._Stuart__inv_matrix, self.model.coordinates_upper_platform).reshape(-1, 3)

# 1 вектор размерностью 24
        self.results = self.results.reshape(-1, 24)

        self.x = self.results[:, :6]

        self.y = self.results[:, 6:]

        columns_my = ['Угол поворота Х', 'Угол поворта Y', 'Угол поворота Z',
                      'Перемещение по Х', 'Перемещение по Y', 'Перемещение по Z',
                      'Длинна 1 ноги', 'Длинна 2 ноги', 'Длинна 3 ноги', 'Длинна 4 ноги', 'Длинна 5 ноги', 'Длинна 6 ноги',
                      'Угол поворота 1 ноги у нижней платформы', 'Угол поворота 1 ноги у верхней платформы', 'Угол поворота 2 ноги у нижней платформы', 'Угол поворота 2 ноги у верхней платформы', 'Угол поворота 3 ноги у нижней платформы', 'Угол поворота 3 ноги у верхней платформы',
                      'Угол поворота 4 ноги у нижней платформы', 'Угол поворота 4 ноги у верхней платформы', 'Угол поворота 5 ноги у нижней платформы', 'Угол поворота 5 ноги у верхней платформы', 'Угол поворота 6 ноги у нижней платформы', 'Угол поворота 6 ноги у верхней платформы']

        if pandas_df:
            return pd.DataFrame(self.results, columns=columns_my)
        else:
            return self.x, self.y
