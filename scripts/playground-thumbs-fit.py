"""Frame the playground gallery thumbnails: trim the white border, centre the picture on a 16:10 white card,
save at 480x300. Run after scripts/playground-thumbs.mjs (needs Pillow)."""
import glob
import os
import sys

from PIL import Image, ImageChops

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '../src/assets/playground')
for f in glob.glob(os.path.join(root, '*/*.webp')):
    im = Image.open(f).convert('RGB')
    box = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).convert('L').point(lambda v: 255 if v > 60 else 0).getbbox()
    if not box:
        continue
    pad = 28
    im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
    w, h = im.size
    tw, th = (w, round(w / 1.6)) if w / h >= 1.6 else (round(h * 1.6), h)
    card = Image.new('RGB', (tw, th), (255, 255, 255))
    card.paste(im, ((tw - w) // 2, (th - h) // 2))
    card.resize((480, 300), Image.LANCZOS).save(f, quality=82, method=6)
