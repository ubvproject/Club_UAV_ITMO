#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практикум 2 — «Пиксели, пороги и цвет». UbV Полигон, занятие 28.10.26.

Кадр — это массив numpy: img[y, x] = [B, G, R], 0…255. Серый кадр — один канал. Почти любое «зрение» начинается с
вопроса «какие пиксели мне нужны?» — порог по яркости или по цвету, — а потом пиксели собирают в контуры.

Сцена-заготовка: пол с неравномерным светом (окно слева, тень справа), оранжевая цель, синий квадрат, ArUco-метка.
    python3 p2_pikseli_i_cvet.py             проверка + картинки в out/p2_*.png
    python3 p2_pikseli_i_cvet.py --bort      НА БОРТУ: три порога на кадре своей камеры + цветной предмет в метрах

Режим --bort: борт в руках камерой вниз над меткой (моторы не запускаются). 20 секунд скрипт считает, в какой доле кадров
метка читается после каждого из трёх порогов. Повторите опыт: (1) ровный свет; (2) тень руки на половину метки;
(3) фонарик телефона сбоку. Картинка 2 × 2 — в out/live.jpg. С --hsv 5,120,80,22,255,255 ищется цветной предмет,
и его положение переводится в сантиметры по лазеру (как в практикуме 5, но «плоско»).
"""
import argparse

import cv2
import numpy as np

from ubv_cv import Aruco, Bort, Score, close, marker_image, out, put

TARGET = (430, 260, 46)              # оранжевая цель: центр x, y и радиус, пикс (для проверки)
ORANGE_LO, ORANGE_HI = (5, 120, 80), (22, 255, 255)   # HSV-порог оранжевого (H в OpenCV: 0…179!)


def scene():
    """Синтетический кадр 640 × 400: свет падает слева, справа — тень, есть шум."""
    h, w = 400, 640
    xx = np.linspace(0, 1, w)[None, :]
    light = 235 - 150 * xx ** 1.4                                   # окно слева, к правому краю темно
    floor = np.repeat(light, h, axis=0) + np.random.default_rng(1).normal(0, 4, (h, w))
    img = cv2.merge([floor * 0.92, floor * 0.95, floor]).clip(0, 255).astype(np.uint8)
    cv2.circle(img, TARGET[:2], TARGET[2], (40, 120, 245), -1)          # оранжевая цель (BGR!)
    cv2.rectangle(img, (60, 250), (150, 340), (200, 90, 30), -1)        # синий квадрат
    m = marker_image(7, 120)
    m = cv2.copyMakeBorder(m, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    shade = np.linspace(1.0, 0.3, m.shape[1])[None, :]               # метка тоже в тени справа
    m = (m * shade).astype(np.uint8)
    img[40:40 + m.shape[0], 470:470 + m.shape[1]] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return img


# ---------------------------------------------------------------------------- A. порог яркости

def porog_fix(gray, t=128):
    """Глобальный порог: пиксель белый (255), если ярче t, иначе чёрный (0). Подсказка: cv2.threshold."""
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return binary


def porog_otsu(gray):
    """Порог Оцу: t выбирается сам по гистограмме. Подсказка: флаг cv2.THRESH_OTSU."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def porog_adapt(gray, block=61, c=10):
    """Адаптивный порог: каждый пиксель сравнивается со средним по окрестности block × block.
    Так работает первый шаг детектора ArUco. Подсказка: cv2.adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C.
    Окно block (нечётное) должно быть больше самого крупного чёрного пятна — иначе его середина «побелеет»."""
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block, c
    )
    return binary


# ---------------------------------------------------------------------------- B. цвет
def maska_hsv(img, lo, hi):
    """Маска пикселей, чей цвет в HSV попадает в [lo, hi]. Зачем HSV: тон (H) почти не меняется от освещения,
    а в BGR тень меняет все три канала. Подсказка: cv2.cvtColor(..., COLOR_BGR2HSV), cv2.inRange."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    return mask


def chistka(mask, k=5):
    """Убрать одиночные шумовые пиксели и дырки: морфологическое «открытие», потом «закрытие».
    Подсказка: cv2.morphologyEx, MORPH_OPEN / MORPH_CLOSE, ядро np.ones((k, k), np.uint8)."""
    kernel = np.ones((k, k), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cleaned


def centr(mask):
    """Центр и площадь самого большого пятна маски -> (x, y, площадь_пикс) или None.
    Подсказка: cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE), max по cv2.contourArea, cv2.moments:
    x = m10 / m00, y = m01 / m00."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    m = cv2.moments(contour)

    if m["m00"] == 0:
        return None

    x = m["m10"] / m["m00"]
    y = m["m01"] / m["m00"]

    return x, y, area


# ---------------------------------------------------------------------------- проверка
def check():
    s = Score("Практикум 2 · пиксели, пороги и цвет")
    img = scene()
    cv2.imwrite(out("p2_scene.png"), img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  кадр: shape = {img.shape}, dtype = {img.dtype}; пиксель [200, 100] (y, x) = {img[200, 100]} (B, G, R)")
    det = Aruco()

    def found(binary):
        """Видит ли детектор метку на картинке после порога (сам детектор порогует ещё раз — это просто тест)."""
        _, ids, _ = det.detect(binary)
        return ids is not None and 7 in ids.flatten()

    def a1():
        b = porog_fix(gray)
        cv2.imwrite(out("p2_porog_fix.png"), b)
        assert b.shape == gray.shape and set(np.unique(b)) <= {0, 255}, "нужна бинарная картинка 0/255"
        return f"метка после порога {'видна' if found(b) else 'НЕ видна'} — посмотрите out/p2_porog_fix.png"

    def a2():
        b = porog_otsu(gray)
        cv2.imwrite(out("p2_porog_otsu.png"), b)
        assert set(np.unique(b)) <= {0, 255}, "нужна бинарная картинка 0/255"
        return f"метка {'видна' if found(b) else 'НЕ видна'} — out/p2_porog_otsu.png"

    def a3():
        b = porog_adapt(gray)
        cv2.imwrite(out("p2_porog_adapt.png"), b)
        assert found(b), "после адаптивного порога метка должна читаться (block нечётный и больше двух клеток метки, c > 0)"
        return "метка видна в тени — out/p2_porog_adapt.png"

    def b1():
        m = chistka(maska_hsv(img, ORANGE_LO, ORANGE_HI))
        cv2.imwrite(out("p2_maska.png"), m)
        r = centr(m)
        assert r is not None, "пятно не найдено"
        x, y, area = r
        close(x, TARGET[0], 2.0, "x центра")
        close(y, TARGET[1], 2.0, "y центра")
        close(area, np.pi * TARGET[2] ** 2, 0.08 * np.pi * TARGET[2] ** 2, "площадь")
        vis = img.copy()
        cv2.drawMarker(vis, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.imwrite(out("p2_cel.png"), vis)
        return f"цель в ({x:.1f}, {y:.1f}), площадь {area:.0f} пикс; радиус ≈ {np.sqrt(area / np.pi):.1f}"

    def b2():
        lo, hi = (100, 120, 60), (130, 255, 255)
        r = centr(chistka(maska_hsv(img, lo, hi)))
        assert r is not None, "синий квадрат не найден — подберите HSV-порог синего"
        close(r[0], 105, 3, "x центра"), close(r[1], 295, 3, "y центра")
        return f"синий квадрат в ({r[0]:.0f}, {r[1]:.0f})"

    s.run("A1 глобальный порог 128", a1)
    s.run("A2 порог Оцу", a2)
    s.run("A3 адаптивный порог", a3)
    s.run("B1 оранжевая цель: маска -> чистка -> центр", b1)
    s.run("B2 синий квадрат тем же кодом", b2)
    s.summary()
    print("\nВопросы: почему глобальный порог «съел» метку в тени? Что будет с HSV-маской под жёлтой лампой?\n"
          "Где на дроне пригодится centr(): посадка на цветную площадку, ворота, «линия» на полу.")


# ---------------------------------------------------------------------------- борт
def bort(seconds, block, hsv):
    import time
    b = Bort()
    det = Aruco()
    names = ("kadr", "fix 128", "otsu", f"adapt {block}")
    hits, n, last, todo = [0, 0, 0, 0], 0, 0.0, set()
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            img = b.kadr()
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            views = [gray]
            for i, fn in enumerate((lambda g: porog_fix(g), porog_otsu, lambda g: porog_adapt(g, block)), 1):
                try:
                    views.append(fn(gray))
                except NotImplementedError:
                    todo.add(names[i])
                    views.append(np.zeros_like(gray))
            n += 1
            tiles = []
            for i, (name, v) in enumerate(zip(names, views)):
                ids = det.detect(v)[1]
                hits[i] += ids is not None
                t = cv2.cvtColor(cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_GRAY2BGR)
                put(t, f"{name}: {'ARUCO OK' if ids is not None else 'no marker'}", (12, 30), 0.7,
                    (80, 220, 80) if ids is not None else (60, 60, 230))
                tiles.append(t)
            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            if hsv:
                h = b.vysota_kamery()
                try:
                    r = centr(chistka(maska_hsv(img, hsv[:3], hsv[3:])))
                except NotImplementedError:
                    r = None
                if r and h:
                    k = h / b.cam_dict["f"]                       # метров на пиксель
                    dx, dy = (r[0] - b.w_px / 2) * k, -(r[1] - b.h_px / 2) * k
                    msg = f"предмет: {dx * 100:+.0f} см вправо, {dy * 100:+.0f} см вперёд от камеры (h {h:.2f} м)"
                    cv2.drawMarker(grid, (int(r[0] / 2), int(r[1] / 2)), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
                    if time.time() - last > 0.5:
                        last = time.time()
                        print(" ", msg)
            b.pokazat(grid)
    except KeyboardInterrupt:
        pass
    finally:
        b.close()
    if not n:
        print("кадров нет")
        return
    print(f"\nМетка читается (из {n} кадров):")
    for name, k in zip(names, hits):
        print(f"  {name:10s} " + ("TODO — допишите функцию" if name in todo else f"{100 * k / n:5.0f} %"))
    print("Запишите в бланк для каждого опыта: ровный свет / тень руки / фонарик.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bort", action="store_true")
    ap.add_argument("--sek", type=float, default=20)
    ap.add_argument("--block", type=int, default=61, help="окно адаптивного порога, нечётное")
    ap.add_argument("--hsv", type=lambda t: [int(x) for x in t.split(",")], default=None,
                    help="H,S,V нижний и верхний, через запятую: 5,120,80,22,255,255")
    a = ap.parse_args()
    bort(a.sek, a.block | 1, a.hsv) if a.bort else check()
