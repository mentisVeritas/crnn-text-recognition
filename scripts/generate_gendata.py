from pathlib import Path
import random
import shutil
import re
import math
import time
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
from faker import Faker

# ---------- CONFIG ----------
OUT_DIR = Path("../data/processed/gendata")
IMG_DIR = OUT_DIR / "images"
FONTS_DIR = Path("./fonts")
COUNT = 100_000

fake = Faker()

# ---------- GLOBAL FONT CACHE ----------
FONT_CACHE = None


# ---------- TEXT ----------
def generate_text():
    mode = random.choices(
        [
            "lower",
            "upper",
            "mixed",
            "numeric",
            "alnum",
            "short",
            "repeat",
            "confusion",
        ],
        weights=[32, 12, 24, 10, 8, 5, 4, 5],
        k=1,
    )[0]

    punctuation = list(".,!?;:-()[]{}<>\"'@#$%^&*_+=/\\|~`&")

    if mode == "lower":
        text = fake.sentence(nb_words=random.randint(2, 10)).lower()

    elif mode == "upper":
        text = fake.sentence(nb_words=random.randint(2, 8)).upper()

    elif mode == "mixed":
        words = fake.words(nb=random.randint(2, 8))

        result = []
        for w in words:
            r = random.random()

            if r < 0.2:
                result.append(w.upper())
            elif r < 0.45:
                result.append(w.capitalize())
            else:
                result.append(w.lower())

        text = " ".join(result)

    elif mode == "numeric":
        patterns = [
            lambda: str(random.randint(0, 99999999)),
            lambda: f"{random.randint(10,99)}:{random.randint(10,59):02d}",
            lambda: f"{random.randint(1,999)}.{random.randint(0,99):02d}",
            lambda: f"+998 {random.randint(10,99)} {random.randint(100,999)} {random.randint(10,99)} {random.randint(10,99)}",
            lambda: f"{random.randint(1000,9999)}-{random.randint(100,999)}",
        ]
        text = random.choice(patterns)()

    elif mode == "alnum":
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        chunks = []
        for _ in range(random.randint(1, 4)):
            chunk = "".join(random.choices(chars, k=random.randint(2, 8)))
            chunks.append(chunk)

        text = " ".join(chunks)

    elif mode == "short":
        variants = [
            fake.word(),
            fake.word().upper(),
            random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
            random.choice(punctuation),
            fake.word()[: random.randint(2, 4)],
        ]
        text = random.choice(variants)

    elif mode == "repeat":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=.#"
        ch = random.choice(chars)
        text = ch * random.randint(3, 12)

    else:
        patterns = [
            "O0O0",
            "Il1I",
            "S5S5",
            "B8B8",
            "rn rn m",
            "0O1Il",
        ]
        text = random.choice(patterns)

    # insert punctuation inside text
    if len(text) > 4 and random.random() < 0.15:
        pos = random.randint(1, len(text) - 2)
        text = text[:pos] + random.choice(punctuation) + text[pos:]

    # wrap punctuation occasionally
    if random.random() < 0.2:
        text = f'"{text}"'

    if random.random() < 0.15:
        text = f'({text})'

    if random.random() < 0.2:
        text += random.choice([".", "!", "?", ":", ";"])

    return re.sub(r"\s+", " ", text).strip()


def generate_punctuation_text():
    symbols = list(".,!?;:-()[]{}<>\"'@#$%^&*_+=/\\|~`")
    length = random.randint(5, 30)

    text = ""
    for _ in range(length):
        if random.random() < 0.8:
            text += random.choice(symbols)
        else:
            text += " "

    return text.strip()


# ---------- FONTS ----------
def load_fonts():
    global FONT_CACHE

    if FONT_CACHE is not None:
        return FONT_CACHE

    if not FONTS_DIR.exists():
        FONT_CACHE = []
        return FONT_CACHE

    FONT_CACHE = list(FONTS_DIR.glob("*.ttf"))
    return FONT_CACHE


def get_font(size):
    fonts = load_fonts()
    if fonts:
        return ImageFont.truetype(str(random.choice(fonts)), size)
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except:
        return ImageFont.load_default()


# ---------- BACKGROUND ----------
def make_background(w, h):
    mode = random.choice([
        "white",
        "noise",
        "gradient",
        "dark",
        "soft_color",
        "ui_dark",
    ])

    if mode == "white":
        return Image.new("RGB", (w, h), (255, 255, 255))

    if mode == "dark":
        val = random.randint(10, 60)
        return Image.new("RGB", (w, h), (val, val, val))

    if mode == "soft_color":
        color = tuple(random.randint(180, 255) for _ in range(3))
        return Image.new("RGB", (w, h), color)

    if mode == "ui_dark":
        palettes = [
            (30, 30, 30),
            (24, 28, 36),
            (15, 20, 30),
            (40, 20, 20),
            (20, 35, 25),
            (20, 20, 45),
        ]
        return Image.new("RGB", (w, h), random.choice(palettes))

    if mode == "noise":
        base = np.random.randint(200, 245, (h, w, 3), dtype=np.uint8)
        noise = np.random.normal(0, 10, (h, w, 3))
        base = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(base)

    base = np.zeros((h, w, 3), dtype=np.uint8)
    c1, c2 = np.random.randint(200, 255), np.random.randint(200, 255)

    for i in range(h):
        val = int(c1 + (c2 - c1) * (i / h))
        base[i, :, :] = val

    return Image.fromarray(base)


# ---------- CURVED TEXT ----------
def draw_curved_text(img, text, font, start_x, base_y):
    draw = ImageDraw.Draw(img)

    bg_mean = np.array(img).mean()
    dark_bg = bg_mean < 128

    # strong readable contrast
    if dark_bg:
        if random.random() < 0.25:
            fill = random.choice([
                (220, 220, 220),
                (86, 156, 214),
                (78, 201, 176),
                (181, 206, 168),
                (255, 220, 120),
            ])
        else:
            val = random.randint(220, 255)
            fill = (val, val, val)
    else:
        if random.random() < 0.25:
            fill = random.choice([
                (0, 0, 0),
                (4, 81, 165),
                (128, 0, 128),
                (163, 21, 21),
                (0, 100, 0),
            ])
        else:
            val = random.randint(0, 40)
            fill = (val, val, val)

    x = start_x
    amplitude = random.uniform(0, 3)
    frequency = random.uniform(0.05, 0.12)

    for i, ch in enumerate(text):
        bbox = font.getbbox(ch)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        offset_y = amplitude * math.sin(i * frequency)
        jitter = random.uniform(-1, 1)

        spacing = random.uniform(0, 1.2)

        pos = (x, base_y + offset_y + jitter)

        # simulate bold/thick fonts
        is_bold = random.random() < 0.2

        if random.random() < 0.25:
            shadow_offset = random.randint(1, 2)
            shadow_color = (0, 0, 0) if dark_bg else (180, 180, 180)
            draw.text(
                (pos[0] + shadow_offset, pos[1] + shadow_offset),
                ch,
                font=font,
                fill=shadow_color,
            )

        stroke_width = 0
        stroke_fill = None

        if random.random() < 0.2:
            stroke_width = random.choice([1, 2])
            stroke_fill = (255, 255, 255) if not dark_bg else (0, 0, 0)

        if is_bold:
            offsets = [
                (0, 0),
                (1, 0),
                (0, 1),
            ]

            if random.random() < 0.5:
                offsets.extend([
                    (1, 1),
                    (-1, 0),
                ])

            for ox, oy in offsets:
                draw.text(
                    (pos[0] + ox, pos[1] + oy),
                    ch,
                    font=font,
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
        else:
            draw.text(
                pos,
                ch,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )

        x += w + spacing


def add_noise(img):
    arr = np.array(img).astype(np.float32)

    noise_type = random.choice(["gaussian", "salt", "speckle"])

    if noise_type == "gaussian":
        arr += np.random.normal(0, random.uniform(3, 10), arr.shape)

    elif noise_type == "salt":
        prob = random.uniform(0.005, 0.02)
        mask = np.random.rand(*arr.shape[:2])
        arr[mask < prob] = 0
        arr[mask > 1 - prob] = 255

    elif noise_type == "speckle":
        arr += arr * np.random.normal(0, 0.2, arr.shape)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # JPEG шум (очень полезно)
    if random.random() < 0.5:
        import io
        buf = io.BytesIO()
        quality = random.randint(35, 85)
        img.save(buf, format="JPEG", quality=quality)
        img = Image.open(buf)

    return img


# ---------- IMAGE ----------
def render(text):
    font_size = random.randint(18, 48)
    font = get_font(font_size)

    left, top, right, bottom = font.getbbox(text)
    w = right - left
    h = bottom - top

    pad_x, pad_y = random.randint(30, 50), random.randint(25, 40)

    img = make_background(w + pad_x * 2, h + pad_y * 2)

    draw_curved_text(img, text, font, pad_x - left, pad_y - top)

    # --- защита от обрезания ---
    bg_fill = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
    img = ImageOps.expand(img, border=25, fill=bg_fill)

    # --- геометрия ---
    angle = random.uniform(-2, 2)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_fill)

    shear = random.uniform(-0.04, 0.04)
    w, h = img.size
    xshift = abs(shear) * h

    img = img.transform(
        (w + int(xshift), h),
        Image.AFFINE,
        (1, shear, -xshift if shear > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC,
        fillcolor=bg_fill,
    )

    # --- перспектива ---
    dx = random.uniform(-0.02, 0.02) * w
    dy = random.uniform(-0.02, 0.02) * h

    coeffs = (1, dx / w, 0, dy / h, 1, 0, 0, 0)
    img = img.transform(img.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    # --- деградация ---
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.0))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    if random.random() < 0.4:
        orig_w, orig_h = img.size

        scale = random.uniform(0.5, 0.9)
        down_w = max(1, int(orig_w * scale))
        down_h = max(1, int(orig_h * scale))

        img = img.resize((down_w, down_h), Image.BILINEAR)
        img = img.resize((orig_w, orig_h), Image.BICUBIC)

    # --- lighting (simulate real photo conditions) ---
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        gradient = np.linspace(0.8, 1.2, w)
        gradient = np.tile(gradient, (h, 1))

        for c in range(3):
            arr[:, :, c] *= gradient

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # --- safe center crop ---
    crop_margin_x = random.randint(0, 3)
    crop_margin_y = random.randint(0, 2)

    img = img.crop((
        crop_margin_x,
        crop_margin_y,
        img.width - crop_margin_x,
        img.height - crop_margin_y,
    ))

    # lighting + noise
    # (lighting inserted above)
    if random.random() < 0.7:
        img = add_noise(img)

    return img


# ---------- MAIN ----------
def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = OUT_DIR / "labels.txt"

    with open(labels_path, "w", encoding="utf-8") as f:
        start_time = time.time()

        progress_bar = tqdm(
            range(COUNT),
            desc="Generating",
            unit="img",
            dynamic_ncols=True,
        )

        for i in progress_bar:
            # --- текст ---
            if random.random() < 0.05:
                text = generate_punctuation_text()
            else:
                text = generate_text()

            text = text[:80]

            # --- изображение ---
            img = render(text)

            name = f"{i:06d}.jpg"
            img.save(IMG_DIR / name, format="JPEG", quality=70, subsampling=2)

            # --- запись label ---
            f.write(f"{name}\t{text}\n")

            progress_bar.set_postfix({
                "last": text[:25],
                "size": f"{img.width}x{img.height}",
            })


            # --- освобождение памяти ---
            img.close()
            del img

    print("DONE:", COUNT)


if __name__ == "__main__":
    main()
