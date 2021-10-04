from PIL import Image
from pathlib import Path

def crop_image(img_path, save=False):
    img = Image.open(img_path)

    w, h = img.width, img.height
    # x0, y0, x1, y1


    # offset = {"1": 0, "2": 0.02, "3": 0.04, "4": 0.06}
    # num = img_path.stem[0]

    if "_100_" in img_path.stem:
        dw = 0.125# + offset[num]
        dh = 0
    elif "_110_" in img_path.stem:
        dw = 0# offset[num]
        dh = 0
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


def concat_images(image_paths: list):



    images = [crop_image(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    offset = 0
    for im in images:
        new_im.paste(im, (offset, 0))
        offset += im.size[0]

    new_im.show()




def main():

    # img_path = Path("figs/1_n90_r40_100_frame0.png")
    # crop_image(img_path)


    src_dir = Path("figs")
    # count = 0
    # for i in range(1, 5)
    i = 1
    # image_paths = [f for f in src_dir.iterdir() if f.suffix == ".png" and "TEST" in f.stem and "new" not in f.stem]# and f.stem == str(i)]
    # image_paths = [f for f in src_dir.iterdir() if f.suffix == ".png" and "TEST" in f.stem and "new" not in f.stem]# and f.stem == str(i)]
    # print(image_paths)
    for f in src_dir.iterdir():
        # if f.suffix == ".png" and "TEST" in f.stem and "new" not in f.stem and f.stem == str(i):
        if f.suffix == ".png" and "cropped" not in f.stem:
            if f.stem[0] == "4":
                crop_image(f, True)



    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    main()
