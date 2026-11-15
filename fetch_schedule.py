#!/usr/bin/env python3
"""Завантажує офіційну картинку графіка Чернігівобленерго,
розпізнає інтервали для однієї черги і оновлює schedule.json.

Запуск:  python fetch_schedule.py [--image local.jpg] [--queue 6.1]
Код виходу 1 = не вдалося впевнено розпізнати (schedule.json не чіпаємо).
"""
import argparse
import datetime
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytesseract

IMAGE_URL = "https://chernihivoblenergo.com.ua/files/other/schedule_groups.jpg"
QUEUES = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2",
          "4.1", "4.2", "5.1", "5.2", "6.1", "6.2"]
OUT = "schedule.json"
DEBUG_DIR = "debug"
KEEP_DAYS = 4


def fail(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)


def download(path):
    req = urllib.request.Request(
        IMAGE_URL,
        headers={"User-Agent": "home-schedule-sync/1.0 (personal ESP32 display)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.URLError as e:
        # сайт віддає неповний ланцюжок сертифікатів; дані публічні й
        # проходять жорстку валідацію нижче, тож повторюємо без перевірки
        print("WARN: TLS-перевірка не пройшла (%s), повтор без неї" % e,
              file=sys.stderr)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            data = r.read()
    if len(data) < 100_000:
        fail("картинка підозріло мала: %d байт" % len(data))
    with open(path, "wb") as f:
        f.write(data)


def save_debug(name, img):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, name), img)


def find_tiles(img):
    """12 персикових плиток -> список (x, y, w, h) у порядку QUEUES."""
    lower = np.array([130, 170, 235])   # BGR
    upper = np.array([195, 225, 255])
    mask = cv2.inRange(img, lower, upper)
    save_debug("mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 250 <= w <= 450 and 250 <= h <= 450:
            tiles.append((x, y, w, h))

    if len(tiles) != 12:
        fail("знайдено %d плиток замість 12 (змінилась розкладка?)" % len(tiles))

    tiles.sort(key=lambda t: t[1] + t[3] / 2)       # за y
    rows = [sorted(tiles[i:i + 3], key=lambda t: t[0])
            for i in range(0, 12, 3)]
    return [t for row in rows for t in row]


def binarize(gray, thr=120, scale=2):
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return cv2.resize(bw, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)


def ocr(img, whitelist, psm):
    cfg = "--psm %d -c tessedit_char_whitelist=%s" % (psm, whitelist)
    return pytesseract.image_to_string(img, config=cfg)


def read_date(img):
    h, w = img.shape[:2]
    roi = img[int(h * 0.058):int(h * 0.093), int(w * 0.36):int(w * 0.66)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 25, 25, 25, 25,
                              cv2.BORDER_CONSTANT, value=255)
    save_debug("date.png", gray)

    votes = {}
    total = 0
    for thr in (100, 120, 140, 160, 180):
        for scale in (2, 3):
            for psm in (7, 13):
                txt = ocr(binarize(gray, thr, scale), "0123456789.", psm)
                total += 1
                m = re.search(r"(d{2}).(d{2}).(20d{2})", txt)
                if not m:
                    continue
                d, mo, y = map(int, m.groups())
                try:
                    key = datetime.date(y, mo, d)
                except ValueError:
                    continue
                votes[key] = votes.get(key, 0) + 1

    print("Голоси за дату:", {k.isoformat(): v for k, v in votes.items()},
          "з", total)
    if not votes:
        fail("не вдалося розпізнати дату на картинці")

    best, n = max(votes.items(), key=lambda kv: kv[1])
    if n < 0.6 * sum(votes.values()) or n < 4:
        fail("дата розпізнана неоднозначно")

    today = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=2))).date()
    if abs((best - today).days) > 2:
        fail("дата на картинці %s надто далека від сьогодні" % best)
    return best


def parse_intervals(raw):
    pat = r"(\d{1,2})\s*[:.]\s*(\d{2})\s*[-–—]\s*(\d{1,2})\s*[:.]\s*(\d{2})"
    out = []
    for h1, m1, h2, m2 in re.findall(pat, raw):
        h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
        if m1 not in (0, 30) or m2 not in (0, 30) or h1 > 23 or h2 > 24:
            return None
        s = h1 * 60 + m1
        e = (h2 * 60 + m2) or 1440
        if e <= s or e > 1440:
            return None
        out.append("%02d:%02d-%02d:%02d" % (h1, m1, h2, m2))
    return out


def read_tile(img, tile, name):
    x, y, w, h = tile
    roi = img[y:y + h, x:x + w]

    body = roi[int(h * 0.28):int(h * 0.96), int(w * 0.20):int(w * 0.98)]
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    save_debug("tile_%s.png" % name, gray)

    best = None
    for thr in (120, 100, 140):
        txt = ocr(binarize(gray, thr), "0123456789:- ", 6)
        digits = re.sub(r"\D", "", txt)
        if not digits:
            best = best or []          # порожня плитка = без відключень
            continue
        iv = parse_intervals(txt)
        if iv:
            return iv
    if best is not None:
        return best
    return None


def label_check(img, tile, expected):
    """М'яка перевірка підпису черги (шрифт стилізований, лише лог)."""
    x, y, w, h = tile
    lab = img[y:y + int(h * 0.26), x:x + w]
    gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    txt = ocr(binarize(gray, 120), "0123456789.", 7).strip()
    print("  label %s -> OCR '%s'" % (expected, txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--queue", default="6.1")
    args = ap.parse_args()

    if args.queue not in QUEUES:
        fail("невідома черга " + args.queue)

    path = args.image
    if not path:
        path = "schedule_groups.jpg"
        download(path)

    img = cv2.imread(path)
    if img is None:
        fail("не вдалося відкрити зображення")

    day = read_date(img)
    print("Дата на картинці:", day.isoformat())

    tiles = find_tiles(img)
    idx = QUEUES.index(args.queue)
    label_check(img, tiles[idx], args.queue)

    iv = read_tile(img, tiles[idx], args.queue)
    if iv is None:
        fail("не вдалося розпізнати інтервали черги " + args.queue)

    text = ", ".join(iv) if iv else "-"
    print("Черга %s: %s" % (args.queue, text))

    data = {"queue": args.queue, "days": {}}
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("queue") == args.queue:
                data["days"] = dict(old.get("days", {}))
        except Exception:
            pass

    data["days"][day.isoformat()] = text
    keys = sorted(data["days"])[-KEEP_DAYS:]
    data["days"] = {k: data["days"][k] for k in keys}
    data["updated"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # не міняти файл, якщо зміст той самий (щоб не плодити коміти)
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("days") == data["days"] and old.get("queue") == data["queue"]:
                print("Без змін")
                return
        except Exception:
            pass

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("schedule.json оновлено")


if __name__ == "__main__":
    main()
