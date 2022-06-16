import numpy as np


def rotx(alfa, matrix=np.eye(4)) -> np.array:
    # Функция принимает на вход угол поворота относительно Х, а так же необязательную матрицу. В итоге получаем матрицу поворота относительно Х
    # на угол alfa размерностью (4,4).
    cos_sin = [np.cos((alfa/180)*np.pi),np.sin((alfa/180)*np.pi)]
    return np.dot(np.array([[1, 0, 0, 0],
                            [0, cos_sin[0], -
                            cos_sin[1], 0],
                            [0, cos_sin[1],
                            cos_sin[0], 0],
                            [0, 0, 0, 1]]), matrix)


def roty(betta, matrix=np.eye(4)) -> np.array:
    # Функция принимает на вход угол поворота относительно Y, а так же необязательную матрицу. В итоге получаем матрицу поворота относительно Y
    # на угол betta размерностью (4,4).
    cos_sin = [np.cos((betta/180)*np.pi),np.sin((betta/180)*np.pi)]
    return np.dot(np.array([[cos_sin[0], 0, cos_sin[1], 0],
                            [0, 1, 0, 0],
                            [-cos_sin[1], 0,
                            cos_sin[0], 0],
                            [0, 0, 0, 1]]), matrix)


def rotz(gamma, matrix=np.eye(4)) -> np.array:
    # Функция принимает на вход угол поворота относительно Z, а так же необязательную матрицу. В итоге получаем матрицу поворота относительно Z
    # на угол gamma размерностью (4,4).
    cos_sin = [np.cos((gamma/180)*np.pi),np.sin((gamma/180)*np.pi)]
    return np.dot(np.array([[cos_sin[0], -cos_sin[1], 0, 0],
                            [cos_sin[1],
                            cos_sin[0], 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]), matrix)


def moving(x:int or float, y:int or float, z:int or float) -> np.array:
    # Функция принимает на вход координаты смещения по осям x,y,z. В итоге получем матрицу перемещения по осям x,y,z размерностью (4,4).
    #
    return np.array([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])


def dob(matrix: np.array) -> np.array:
    # Функция которая принимает на вход координаты верхней платформы размерностью (1,3) (строка), чтобы превратить ее в матрицу столбец с добавленной
    # 1 в конце для перемножения матриц. Итог матрица - столбец координат с добавленной 1 в конце размерностью (1,4).
    return np.append(matrix, 1).reshape(-1, 1)


def len_leg(xyz1: np.array, xyz0: np.array) -> np.array:
    # Функция выполняет расчет скалярного произведения(длинны вектора). Итог длинна прямой между 2 точками.
    return np.sqrt(np.sum(np.multiply((xyz1 - xyz0), (xyz1 - xyz0))))


def calculation_new_coordinates(Peremesh_Povorot: np.array, coordinates: np.array) -> np.array:
    # Расчет координат при повороте и перемещении. 1 аргумент матрица поворота и перемещения размерность (4,4). 2 аргумент координаты,
    #  матрица или строка типа list размерностью (1,3)

    try:

        new_coordinates = np.array([])

        for coordinate in coordinates:

            new_coordinates = np.append(
                new_coordinates, np.delete((Peremesh_Povorot@dob(coordinate)), -1))
        return new_coordinates.reshape(-1,3)

    except ValueError:

        return np.delete((Peremesh_Povorot@dob(coordinates)), -1)


def transformation(alfa:int or float, betta:int or float, gamma:int or float, x: int or float, y: int or float, z:int or float) -> np.array:

    return moving(x, y, z)@rotx(alfa)@roty(betta)@rotz(gamma)


def angle_between_vectors(coordinate_upper_leg:np.array, coordinate_lower_leg: np.array, lower_center_coordinate:np.array, upper_center_coordinate:np.array) -> np.ndarray:

    vector1_upper = upper_center_coordinate - coordinate_upper_leg

    vector2_upper = coordinate_lower_leg - coordinate_upper_leg

    vector1_lower = lower_center_coordinate - coordinate_lower_leg

    vector2_lower = coordinate_upper_leg - coordinate_lower_leg

    angle = np.array([vector1_upper.dot(vector2_upper) / (np.linalg.norm(vector1_upper) * np.linalg.norm(vector2_upper)),
                      vector1_lower.dot(vector2_lower) / (np.linalg.norm(vector1_lower) * np.linalg.norm(vector2_lower))])

    return np.arccos(angle) * 180/np.pi
