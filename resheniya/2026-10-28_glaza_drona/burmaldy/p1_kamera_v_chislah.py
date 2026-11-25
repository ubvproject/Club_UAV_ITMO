#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практикум 1 — «Камера в числах». UbV Полигон, занятие 28.10.26.

Камера-обскура: точка (X, Y, Z) перед камерой попадает в пиксель u = f·X/Z + cx, v = f·Y/Z + cy.
f — фокусное расстояние В ПИКСЕЛЯХ (f_мм / размер_пикселя_мм). Всё остальное выводится из одной этой формулы.

Задание: допишите функции с TODO (их 7), запустите — увидите проверку и таблицу для Пионера.
    python3 p1_kamera_v_chislah.py               проверка + таблица (ноутбук или борт)
    python3 p1_kamera_v_chislah.py --bort        НА БОРТУ: фокус своей камеры по метке 14,5 см и лазеру

Режим --bort: метка id 15 на полу, борт в руках камерой вниз (пропеллеры сняты, моторы не запускаются).
Держите борт ровно, метку — в центре кадра; поднимайте от 0,3 до 1,5 м. Скрипт печатает высоту по лазеру,
сторону метки в пикселях и фокус f = s·h/S. В конце — медиана f и кадр на полу с вашим f.
"""
import argparse
import math

import numpy as np

from ubv_cv import MARKER, PIONEER, SIM_CAM, Aruco, Bort, Score, close, out, put, side_px

def fov_deg(size_px, f_px):
    return 2 * math.atan(size_px / (2 * f_px)) * 180 / math.pi
"""Угол обзора (град) по стороне кадра size_px при фокусе f_px.
    Подсказка: половина кадра size_px/2 видна под углом atan((size_px/2) / f)."""

def footprint(h, cam):
    width = h * cam["w"] / cam["f"]
    height = h * cam["h"] / cam["f"]
    return width, height
    """Сколько метров пола видно в кадре с высоты h (камера смотрит отвесно вниз) -> (ширина, высота).
    Подсказка: подобные треугольники — ширина_м / h = ширина_пикс / f."""

def gsd(h, cam):
    return h / cam["f"]
"""Сколько метров пола приходится на один пиксель с высоты h (GSD — ground sample distance), м/пикс."""

def marker_px(side_m, h, cam):
    return cam["f"] * side_m / h
"""Длина стороны метки side_m в пикселях с высоты h."""
def blur_px(v, t_exp, h, cam):
    return cam["f"] * v * t_exp / h

def max_height(side_m, min_px, cam):
    return cam["f"] * side_m / min_px
"""С какой наибольшей высоты метка ещё будет не меньше min_px пикселей?"""
def fokus_po_metke(side_px, h, side_m):
    return side_px * h / side_m

# ---------------------------------------------------------------------------- проверка
def check():
    s = Score("Практикум 1 · камера в числах")
    c = dict(w=1000, h=500, f=500.0)
    s.run("fov_deg: кадр 1000 пикс при f = 500 -> 90°", lambda: close(fov_deg(1000, 500), 90.0, 0.01, "угол"))
    s.run("footprint: c 1 м видно 2 × 1 м", lambda: (close(footprint(1.0, c)[0], 2.0, 1e-6, "ширина"),
                                                    close(footprint(1.0, c)[1], 1.0, 1e-6, "высота")) and None)
    s.run("gsd: с 2 м пиксель = 4 мм", lambda: close(gsd(2.0, c), 0.004, 1e-9, "gsd"))
    s.run("marker_px: метка 0,1 м с 1 м = 50 пикс", lambda: close(marker_px(0.1, 1.0, c), 50.0, 1e-6, "сторона"))
    s.run("blur_px: 0,5 м/с × 20 мс с 1 м = 5 пикс", lambda: close(blur_px(0.5, 0.02, 1.0, c), 5.0, 1e-6, "смаз"))
    s.run("max_height: 0,1 м ≥ 25 пикс -> 2 м", lambda: close(max_height(0.1, 25, c), 2.0, 1e-6, "высота"))
    s.run("fokus_po_metke: 0,1 м = 50 пикс с 1 м -> 500", lambda: close(fokus_po_metke(50, 1.0, 0.1), 500.0, 1e-6, "f"))
    return s.summary()


def table():
    for cam in (PIONEER, SIM_CAM):
        try:
            fx, fy = fov_deg(cam["w"], cam["f"]), fov_deg(cam["h"], cam["f"])
        except NotImplementedError:
            return
        print(f"\n{cam['name']}: {cam['w']} × {cam['h']}, f = {cam['f']:.0f} пикс, обзор {fx:.1f}° × {fy:.1f}°")
        print("  высота | кадр на полу, м | мм/пикс | метка 14,5 см, пикс | смаз 0,5 м/с · 20 мс, пикс")
        for h in (0.5, 1.0, 1.5, 1.75, 2.0):
            try:
                W, H = footprint(h, cam)
                print(f"  {h:5.2f}  | {W:5.2f} × {H:4.2f}    | {gsd(h, cam) * 1000:6.2f}  | {marker_px(MARKER, h, cam):8.0f}"
                      f"            | {blur_px(0.5, 0.02, h, cam):6.1f}")
            except NotImplementedError:
                print("  (допишите footprint, gsd, marker_px, blur_px)")
                break
    try:
        print(f"\nМетка 14,5 см ≥ 20 пикс на борту — до {max_height(MARKER, 20, PIONEER):.1f} м; ≥ 40 пикс — до "
              f"{max_height(MARKER, 40, PIONEER):.1f} м.")
    except NotImplementedError:
        pass
    print("\nВопросы (ответы — в бланк):\n"
          "  1. Поле 3 × 3 м, метки до ±1,37 м от центра. Сколько снимков нужно с 1,0 м? А с 1,75 м?\n"
          "  2. Почему кадр с разгона даёт смаз, а с висения — нет? Какая скорость даёт смаз 1 пикс на 1,75 м?\n"
          "  3. Камера сима 1280 × 720, борта 1080 × 720, а f пересчитан по ширине. Что из таблицы от этого меняется?")


# ---------------------------------------------------------------------------- борт
def bort(size_m, seconds):
    import statistics as st
    import time
    import cv2
    b = Bort()
    det = Aruco()
    fs, last, line = [], 0.0, ""
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            img = b.kadr()
            if img is None:
                continue
            h_px, w_px = img.shape[:2]
            corners, ids, _ = det.detect(img)
            h = b.vysota_kamery()
            roll, pitch = b.naklon()
            line = f"h {h:.2f} m" if h else "laser -"
            if ids is not None:
                k = max(range(len(ids)), key=lambda i: side_px(corners[i]))
                s_px = side_px(corners[k])
                cx, cy = np.asarray(corners[k]).reshape(4, 2).mean(axis=0)
                ok = (h and abs(roll) < 5 and abs(pitch) < 5
                      and abs(cx - w_px / 2) < w_px / 4 and abs(cy - h_px / 2) < h_px / 4)
                cv2.aruco.drawDetectedMarkers(img, [corners[k]], ids[k:k + 1])
                line += f"  side {s_px:.0f} px"
                if ok:
                    try:
                        f = fokus_po_metke(s_px, h - 0.001, size_m)      # метка толщиной ~1 мм
                        fs.append(f)
                        line += f"  f {f:.0f} px"
                    except NotImplementedError:
                        line += "  (TODO fokus_po_metke)"
                else:
                    line += "  (ровнее и метку в центр)"
            cv2.drawMarker(img, (w_px // 2, h_px // 2), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            put(img, f"{line}  roll {roll:+.0f} pitch {pitch:+.0f}", (20, 40))
            b.pokazat(img)
            if time.time() - last > 0.5:
                last = time.time()
                print(" ", line, f"  крен {roll:+.1f}° тангаж {pitch:+.1f}°")
    except KeyboardInterrupt:
        pass
    finally:
        b.close()
    if not fs and "TODO" in line:
        print("\nДопишите fokus_po_metke() — без неё f не посчитать.")
        return
    if len(fs) < 5:
        print("\nМало замеров f: метка 14,5 см в центре кадра, борт ровно, лазер видит пол.")
        return
    f = st.median(fs)
    nom = b.cam_dict["f"]
    me = dict(b.cam_dict, f=f, name="Пионер · MAIN (мой замер)")
    print(f"\nМой фокус: f = {f:.0f} пикс при ширине {b.w_px} (по {len(fs)} кадрам, разброс ±{st.pstdev(fs):.0f}); "
          f"номинал {nom:.0f} — разница {100 * (f / nom - 1):+.1f} %.")
    print(f"Обзор {fov_deg(b.w_px, f):.1f}° × {fov_deg(b.h_px, f):.1f}°.")
    for h in (1.0, 1.75):
        W, H = footprint(h, me)
        print(f"  с {h:.2f} м: кадр {W:.2f} × {H:.2f} м, метка {marker_px(MARKER, h, me):.0f} пикс")
    import json
    with open(out("bort_p1.json"), "w", encoding="utf-8") as fo:
        json.dump({"f": f, "w": b.w_px, "h": b.h_px, "n": len(fs), "f_nominal": nom}, fo, indent=1)
    print("Сохранено в out/bort_p1.json. Запишите f в бланк — сравним с калибровкой (практикум 4 --bort).\n"
          "Проверка рулеткой: положите рулетку на пол поперёк кадра, на высоте h прочитайте по out/live.jpg ширину L\n"
          f"между краями кадра: f = {b.w_px}·h/L должно совпасть с f по метке.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bort", action="store_true")
    ap.add_argument("--size", type=float, default=MARKER, help="сторона метки, м (чёрный квадрат, линейкой)")
    ap.add_argument("--sek", type=float, default=60, help="сколько секунд мерить")
    a = ap.parse_args()
    if a.bort:
        bort(a.size, a.sek)
    else:
        check()
        table()
