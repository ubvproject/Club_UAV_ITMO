#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Задача занятия 21.10.26 — «Разведка 3 × 3».
Полигон UbV.

1. Взлёт из центра поля (0, 0).
2. Поиск всех 8 меток ArUco.
3. Доклад координат REPORT.
4. Облёт меток по возрастанию ID.
5. Посадка в зоне H и выключение моторов.

Поле: 3 × 3 м.
Камера: MAIN, 1280 × 720.
Словарь: DICT_4X4_50.
Сторона метки: 14,5 см.
Лимит времени: 300 секунд.
"""

import math
import time
import statistics
import cv2
from pioneer_sdk2 import (Camera, CameraType, Pioneer, ServoCamera)

# ============================================================
# 1. ПАРАМЕТРЫ МИССИИ
# ============================================================

FLY_ALT = 1.7           # Высота разведки, м
VISIT_ALT = 0.8         # Высота посещения меток, м
V_MAX = 0.4             # Максимальная горизонтальная скорость
KP = 0.8                # Коэффициент регулятора
FX_1920 = 1166.0        # Фокус для ширины 1920 пикселей
CAM_DY = 0.06           # Камера находится на 6 см впереди центра
SCAN_HOLD = 1.0         # Время стабилизации при разведке
VISIT_HOLD = 1.2        # Время зависания над меткой
MISSION_LIMIT = 300     # Общий лимит времени, секунд
SCAN_POINTS = [(0.0, -0.7), (0.0,  0.7)]  # точки разведки - 2 кадра с перекрытием позволяют увидеть все возможные положения центров меток в пределах поля
GOTO_RADIUS = 0.08  # точность перемещения
MARKER_RADIUS = 0.15  # допустимое расстояние до метки

# Объекты управления
drone = Pioneer()  # экземпляр дрона
camera = Camera(camera_type=CameraType.MAIN)  # основная камера
servo = ServoCamera()  # сервопривод подвеса камеры

# Детектор ArUco
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # загрузка словаря ArUco
aruco = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())  # создание детектора
found = {}  # словарь обнаруженных меток: {ID: список измерений координат}
mission_start = time.time()  # момент старта миссии
airborne = False  # флаг "в воздухе"
landed = False  # флаг "совершил посадку"

# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

# Ограничение значения заданным диапазоном
def clamp(v, a):  # возвращает v, зажатое в [-a, +a]
    return max(-a, min(a, v))

# Получение координат дрона
def pos():  # возвращает (x, y, z) по LPS
    while True:
        p = drone.get_local_position_lps()
        if p is not None:
            return p
        time.sleep(0.02)

# Получение высоты по лазеру
def get_height():
    h = drone.get_dist_sensor_data()
    if h is None:
        return None
    return float(h)

# Остановка горизонтального движения
def stop():  # обнуление всех скоростей
    drone.set_manual_speed(vx=0, vy=0, vz=0, yaw_rate=0)

# Проверка ориентации перед фотографированием, если дрон сильно наклонён, ждём стабилизации
def wait_stable(timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        orientation = drone.get_orientation()
        if orientation is not None:
            roll, pitch, yaw = orientation
            if abs(roll) < 5 and abs(pitch) < 5:
                return True
        time.sleep(0.1)
    print("Предупреждение: дрон не полностью стабилизировался")
    return False

# Проверка лимита времени
def check_time():  # бросает исключение при превышении MISSION_LIMIT
    if time.time() - mission_start > MISSION_LIMIT:
        raise RuntimeError("Превышен лимит времени миссии")

# ============================================================
# 3. ПЕРЕМЕЩЕНИЕ ДРОНА
# ============================================================

# Полёт в точку (x, y, z)
def goto(x, y, z, radius=GOTO_RADIUS, hold=0.5, timeout=25):
    t0 = time.time()  # засекаем время старта
    inside = None   # момент входа в радиус (None - снаружи)
    while time.time() - t0 < timeout:  # пока не истёк таймаут перелёта
        check_time()  # проверка общего лимита миссии
        px, py, pz = pos()  # текущая позиция
        # Ошибка положения
        ex = x - px
        ey = y - py
        ez = z - pz
        distance = math.sqrt(ex**2 + ey**2)  # горизонтальное расстояние до цели
        # Проверяем достижение точки
        if distance < radius and abs(ez) < 0.1:  # если внутри радиуса и по высоте
            inside = inside or time.time()  # фиксируем момент входа (если ещё не было)
            stop()
            if time.time() - inside >= hold:   # если пробыли внутри hold секунд
                return True
        else:
            inside = None
            # П-регулятор
            vx = clamp(KP * ex, V_MAX)
            vy = clamp(KP * ey, V_MAX)
            vz = clamp(0.5 * ez, 0.2)
            drone.set_manual_speed(vx=vx, vy=vy, vz=vz, yaw_rate=0)  # отправка команды скорости
        # Команды отправляются 20 раз в секунду
        time.sleep(0.05)
    stop()
    print(
        f"Таймаут: не удалось достичь "
        f"({x:.2f}, {y:.2f}, {z:.2f})"
    )
    return False

# ============================================================
# 4. ОБНАРУЖЕНИЕ ARUCO-МЕТОК
# ============================================================

# Возвращает [(id, X, Y)] по одному кадру
def see_markers():
    try:  # пытаемся получить кадр
        frame = camera.get_cv_frame()  # кадр с основной камеры (BGR)
    except TimeoutError:
        return []
    if frame is None:
        return []
    px, py, pz = pos()  # координаты дрона
    laser = get_height()  # высота камеры над полом
    if laser is None:  # если нет данных
        return []
    h = laser + 0.02  # поправка +2 см
    if h <= 0:  # защита от деления на ноль
        return []
    H, W = frame.shape[:2]  # размер изображения
    fx = FX_1920 * W / 1920.0  # фокусное расстояние для текущего разрешения
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # перевод изображения в оттенки серого
    # Поиск ArUco
    corners, ids, rejected = aruco.detectMarkers(gray)  # детекция маркеров
    result = []
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            points = corner.reshape(4, 2)  # 4 угла метки
            u, v = points.mean(axis=0)  # центр метки в пикселях
            # Перевод пикселей в метры
            X = px + (u - W / 2) * h / fx
            Y = py + CAM_DY - (v - H / 2) * h / fx
            result.append((int(marker_id), X, Y))
    return result

# ============================================================
# 5. СБОР И ОБРАБОТКА КООРДИНАТ
# ============================================================

# Сбор нескольких кадров в текущей точке
def scan_position():
    wait_stable()
    for frame_number in range(8):
        check_time()
        markers = see_markers()
        for mid, x, y in markers:
            if mid not in found:
                found[mid] = []
            found[mid].append((x, y))
        time.sleep(0.12)

# Медианная координата метки по всем измерениям
def get_marker_position(mid):
    measurements = found.get(mid, [])
    if not measurements:
        return None
    xs = [item[0] for item in measurements]
    ys = [item[1] for item in measurements]
    x = statistics.median(xs)
    y = statistics.median(ys)
    return x, y

# Доклад координат всех найденных меток
def report_markers():
    print("\n========== REPORT ==========")
    for mid in sorted(found):
        position = get_marker_position(mid)
        if position is None:
            continue
        x, y = position
        print(f"REPORT {mid} {x:.2f} {y:.2f}")

# ============================================================
# 6. УТОЧНЕНИЕ ПОЛОЖЕНИЯ МЕТКИ
# ============================================================

# Уточнение координат метки во время посещения
def refine_marker(mid, x, y):
    """
    Повторное обнаружение метки непосредственно во время посещения.
    Если метка видна, её координаты уточняютс по нескольким кадрам.
    """
    measurements = []
    for _ in range(6):
        check_time()
        markers = see_markers()
        for marker_id, mx, my in markers:
            if marker_id == mid:
                measurements.append((mx, my))
        time.sleep(0.12)
    if not measurements:
        print(
            f"Метка {mid}: повторно не обнаружена, "
            f"используем первоначальные координаты"
        )
        return x, y
    new_x = statistics.median([item[0] for item in measurements])
    new_y = statistics.median([item[1] for item in measurements])
    # Добавляем новые измерения
    found[mid].extend(measurements)
    print(
        f"Уточнение метки {mid}: "
        f"({new_x:.2f}, {new_y:.2f})"
    )
    return new_x, new_y

# ============================================================
# 7. ПОСЕЩЕНИЕ МЕТКИ
# ============================================================

# Полный цикл посещения одной метки
def visit_marker(mid):
    position = get_marker_position(mid)
    if position is None:
        return False
    x, y = position
    print(f"\nПосещение метки ID={mid}")
    # Первый подлёт - двухступенчатый:
    # 1) сначала снижаемся на месте до VISIT_ALT
    px, py, pz = pos()
    if not goto(px, py, VISIT_ALT, radius=0.10, hold=0.3):
        return False
    # 2) потом летим горизонтально на этой высоте
    if not goto(x, y, VISIT_ALT, radius=0.08, hold=0.5):
        return False
    wait_stable()
    x, y = refine_marker(mid, x, y) # уточняем координаты по камере
    if not goto(x, y, VISIT_ALT, radius=0.08, hold=0.5):  # повторный подлёт к уточнённым координатам
        return False
    print(f"REPORT {mid} {x:.2f} {y:.2f}")  # повторный доклад: судья учитывает последний
    if not goto(x, y, VISIT_ALT, radius=MARKER_RADIUS, hold=VISIT_HOLD):  # финальное зависание
        return False
    print(f"Метка {mid} посещена")
    return True

# ============================================================
# 8. ПОСАДКА
# ============================================================

# Ручная посадка в текущей точке
def land_here(timeout=30):
    """
    Самостоятельная посадка.
    Дрон удерживает положение по X и Y и постепенно снижается до касания пола.
    Моторы здесь не выключаются.
    """
    print("\nНачинаем посадку")
    x, y, z = pos()
    t0 = time.time()
    while time.time() - t0 < timeout:
        px, py, pz = pos()
        height = get_height()
        if height is None:
            stop()
            print("Ошибка: нет данных лазерного датчика")
            return False
        # Касание пола
        if height <= 0.05:
            stop()
            time.sleep(1.0)
            print("Посадка завершена")
            return True
        # Удержание координат и снижение
        vx = clamp(KP * (x - px), 0.2)
        vy = clamp(KP * (y - py), 0.2)
        drone.set_manual_speed(vx=vx, vy=vy, vz=-0.2, yaw_rate=0)
        time.sleep(0.05)
    stop()
    print("Ошибка: таймаут посадки")
    return False

# ============================================================
# 9. ОСНОВНАЯ МИССИЯ
# ============================================================

# Весь сценарий обёрнут в try/except
try:
    print("\n========== НАЧАЛО МИССИИ ==========\n")
    # Подготовка камеры
    servo.set_angle(-80)
    time.sleep(1.0)
    # Взлёт
    print("Взлёт из зоны H")
    drone.arm()
    drone.takeoff()
    airborne = True
    print("Взлёт выполнен")
    # ЭТАП 1. РАЗВЕДКА
    print("\n========== РАЗВЕДКА ==========\n")
    for index, (cx, cy) in enumerate(SCAN_POINTS):
        check_time()
        print(
            f"Точка разведки {index + 1}: "
            f"({cx:.2f}, {cy:.2f})"
        )
        # Перемещение на высоту 1.7 м
        if not goto(cx, cy, FLY_ALT, hold=SCAN_HOLD):
            raise RuntimeError("Не удалось достичь точки разведки")
        # Стабилизация
        wait_stable()
        # Получение нескольких кадров
        scan_position()
        print(f"Обнаружено различных ID: {len(found)}")
    # ЭТАП 2. ПРОВЕРКА И ДОКЛАД
    print("\n========== РЕЗУЛЬТАТЫ РАЗВЕДКИ ==========\n")
    print(f"Найдено меток: {len(found)} из 8")
    if len(found) < 8:
        print(
            "ВНИМАНИЕ: обнаружены не все метки. "
            "Необходимо проверить результаты разведки."
        )
    report_markers()
    # ЭТАП 3. ОБЛЁТ ПО ВОЗРАСТАНИЮ ID
    print("\n========== ОБЛЁТ МЕТОК ==========\n")
    marker_ids = sorted(found.keys())
    for mid in marker_ids:
        check_time()
        success = visit_marker(mid)
        if not success:
            print(f"Не удалось выполнить посещение метки {mid}")
    # ЭТАП 4. ВОЗВРАЩЕНИЕ В ЦЕНТР
    print("\n========== ВОЗВРАЩЕНИЕ В H ==========\n")
    # Сначала возвращаемся на безопасную высоту
    if not goto(0.0, 0.0, FLY_ALT, hold=0.5):
        raise RuntimeError("Не удалось вернуться в центр поля")
    # Снижаемся до удобной высоты перед посадкой
    if not goto(0.0, 0.0, 0.5, hold=0.5):
        raise RuntimeError("Не удалось занять положение для посадки")
    # ЭТАП 5. ПОСАДКА И ВЫКЛЮЧЕНИЕ МОТОРОВ
    if land_here():
        landed = True
        airborne = False
        # Выключаем моторы только на земле
        drone.disarm()
        print("\nМоторы выключены")
    else:
        raise RuntimeError("Не удалось подтвердить успешную посадку")
    print("\n========== МИССИЯ ЗАВЕРШЕНА ==========\n")

# ============================================================
# 10. ОБРАБОТКА ОШИБОК
# ============================================================

except Exception as error:
    print("\nОШИБКА МИССИИ:", error)
finally:
    # Если произошла ошибка в воздухе, пытаемся выполнить безопасную посадку.
    if airborne and not landed:
        print("\nАварийная попытка посадки")
        try:
            if land_here():
                landed = True
                airborne = False
                drone.disarm()
                print("Моторы выключены после посадки")
            else:
                print(
                    "ВНИМАНИЕ: посадка не подтверждена. "
                    "Моторы не выключаем."
                )
        except Exception as error:
            print("Ошибка при аварийной посадке:", error)
    # Соединение закрываем только после посадки
    if landed:
        try:
            drone.close_connection()
        except Exception as error:
            print("Ошибка закрытия соединения:", error)
