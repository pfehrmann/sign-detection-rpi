from scipy.misc import imsave
from sign_detection.GTSDB.SlidingWindow.batchloader import BatchLoader


def generate_no_sign_images(count_images, path, gtsdb_root, offset=12629):
    params = {'gtsdb_root': gtsdb_root, 'window_size': 64}
    loader = BatchLoader(params, num=100000, fraction=0.5)
    images = loader.generate_windows_without_signs(size=64, regions_per_image=3, max_overlap=0.0, transpose=None)
    out = ""
    for i in range(offset + 1, offset + 1 + count_images):
        while len(images) < 1:
            images = loader.generate_windows_without_signs(size=64, regions_per_image=3, max_overlap=0.0,
                                                           transpose=None)
        imgpath = path + "/" + format(i, '05d') + ".ppm"
        imsave(imgpath, images.pop()[0])
        out += imgpath + " 43\n"
        if i % (count_images / 100) == 0:
            print str(float(i) / float(count_images)) + "% done"

    print out


generate_no_sign_images(40000, "C:/development/GTSRB_AUGMENT/LessClasses/Training/00043",
                        "C:/development/FullIJCNN2013/FullIJCNN2013", offset=0)
# generate_no_sign_images(700, "C:/development/GTSRB_AUGMENT/LessClasses/Validation/00043", "C:/development/FullIJCNN2013/FullIJCNN2013", offset=0)
