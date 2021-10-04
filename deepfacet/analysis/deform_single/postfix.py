from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

def crop_image(img_path, save=False):
    img = Image.open(img_path)

    w, h = img.width, img.height
    # x0, y0, x1, y1


    # offset = {"1": 0, "2": 0.02, "3": 0.04, "4": 0.06}
    # num = img_path.stem[0]

    dh = 0.075
    if "p1_" in img_path.stem:
        dw = 0.00# + offset[num]
    elif "p0_" in img_path.stem:
        dw = 0.25# offset[num]
    else:
        raise NotImplementedError

    box = (w*dw, h*dh, w*(1-dw), h*(1-dh))
    new_img = img.crop(box)

    new_fname = f"{img_path.parent}/cropped_{img_path.stem}.png"
    print(f"cropping {img_path}")
    if save:
        new_img.save(new_fname)
        # new_img.show()

    return new_img


def concat_images(im1, im2, new_name, border_width = None):
    """
    concatenates two images horizontally
    """

    images = [crop_image(x, False) for x in (im1, im2)]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)


    new_im = Image.new('RGBA', (total_width, max_height))


    offset = 0
    for im in images:
        # new_im.paste(im, (offset, dw))
        new_im.paste(im, (offset, 0))
        offset += im.size[0]


    if border_width:
        w = new_im.size[0]-2
        h = new_im.size[1]-1
        draw = ImageDraw.Draw(new_im)
        for i in range(border_width):
            draw.line([(0+i, 0+i), (0+i, h-i)], fill=(0,0,0))
            draw.line([(0+i, 0+i), (w-i, 0+i)], fill=(0,0,0))
            draw.line([(0+i, h-i), (w-i, h-i)], fill=(0,0,0))
            draw.line([(w-i, 0+i), (w-i, h-i)], fill=(0,0,0))

    new_im.show()
    new_im.save(new_name)




def main():


    src_dir = Path("figs")
    i = 1
    # for f in src_dir.iterdir():
    #     # if f.suffix == ".png" and "TEST" in f.stem and "new" not in f.stem and f.stem == str(i):
    #     if f.suffix == ".png" and "cropped" not in f.stem:
    #         crop_image(f, True)

    # for f in src_dir.iterdir():
        # if f.suffix == ".png" and "cropped" in f.stem:
            # con

    im1 = ["p1_frame0.png", "p0_frame0.png"]
    im2 = ["p1_frame906.png", "p0_frame906.png"]
    im3 = ["p1_frame907.png", "p0_frame907.png"]
    im4 = ["p1_frame2000.png", "p0_frame2000.png"]

    i = 0
    for im in (im1, im2, im3, im4):
        d = Path("figs")
        outname = d / f"border_cropped_{i}.png"
        concat_images(d / im[0], d / im[1], outname, 2)
        i += 1
        # concat_images(f"figs/{im[0]}", f"figs/{im[1]}")






if __name__ == '__main__':
    main()
