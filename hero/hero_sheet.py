"""Contact sheet of the title-page candidates: every candidate at one third of its size (1654 x 693 px), stacked, each
under a line with its id, extent and ground pixel size (the only text of the whole set).

usage: python hero_sheet.py <candidate dir> <out.png>          (env agl on s02)"""
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_render import FRAMES, PX                                # noqa: E402

FONT = "/data/befahl/projects/AGL/basodino/hero/fonts/SourceSans3-Semibold.ttf"
W, H, M, T = 1654, 693, 40, 58                                   # image size, margin, label height


def main(d, out):
    font, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT.replace("Semibold", "Regular"), 24)
    S = Image.new("L", (W + 2 * M, M + len(FRAMES) * (T + H + M)), 255)
    dr = ImageDraw.Draw(S)
    for k, (fid, (west, north, res, rot)) in enumerate(FRAMES.items()):
        y = M + k * (T + H + M)
        e, s = west + PX[0] * res, north - PX[1] * res
        dr.text((M, y), fid, fill=30, font=font)
        info = (f"E {west:.0f}-{e:.0f}, N {s:.0f}-{north:.0f}, {res:g} m per pixel"
                + (", turned 180 degrees (south up)" if rot else ""))
        dr.text((M + font.getlength(fid) + 24, y + 5), info, fill=90, font=small)
        S.paste(Image.open(f"{d}/{fid}.png").resize((W, H), Image.LANCZOS), (M, y + T))
    S.save(out, dpi=(150, 150), optimize=True)
    print(out, S.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
