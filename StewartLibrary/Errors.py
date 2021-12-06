import numpy as np


class Check():

    def __checkNumbers(numbers):

        if isinstance(numbers, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
            return numbers

        else:
            raise ValueError('Ошибка задания значений')

    def __checkArrays(array):

        if isinstance(array, list):

            for number in array:

                if Check._Check__checkNumbers(number) == number:
                    continue
                else:
                    raise ValueError(
                        'Одно из значений в массиве не является числом')

            return np.array(array)

        elif isinstance(array, np.ndarray):

            for number in array:

                if Check._Check__checkNumbers(number):
                    continue
                else:
                    raise ValueError(
                        'Одно из значений в массиве не является числом')

            return array

        else:
            raise TypeError('Неправильно задан массив')

    def __checkShape(array):

        if (len(array.shape) == 1 and array.shape[0] == 3):

            return array

        else:
            raise ValueError('Размерность массива не соответствует нужному')

    def __checkNames(platform, system):

        if (platform == 'upper' or platform == 'lower') and (system == 'local' or system == 'global'):

            return True

        else:

            raise KeyError(
                'Неверное название платформы или системы координат')