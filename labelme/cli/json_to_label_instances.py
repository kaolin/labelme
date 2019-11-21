import argparse
import base64
import json
import os
import os.path as osp
import warnings

import numpy as np
import PIL.Image

from labelme import utils


def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a multi-image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('-s', '--scale', default=None)
    parser.add_argument('-clean', action='store_true', help='if specified, will save segmented images with transparent backgrounds')
    args = parser.parse_args()

    json_file = args.json_file

    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            #print("READING {}".format(imagePath))
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    img_pil = PIL.Image.fromarray(img)
    img_shape = img.shape[0:2]
    print('np shape {} vs pil shape {}'.format(img_shape,img_pil.size))
    if args.scale is not None:
        img_pil = img_pil.resize([int(dim * float(args.scale)) for dim in img_shape])
        img = np.array(img_pil)
        img_shape = img.shape[0:2]
    a, b = img_shape
    i = 0
    zpad = len(str(len(data['shapes'])))
    for shape in data['shapes']:
        i += 1
        points = shape['points']
        if len(points) == 1:
            # NAH
            continue
        elif len(points) == 2:
            # make a bbox of the line
            x1, y1 = np.min(points,0)
            x2, y2 = np.max(points,0)
            points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        label = shape['label']
        mask_pil = None
        if args.clean:
            mask = utils.shape.shape_to_mask(img_shape, points, shape.get('shape_type', None))
            mask_pil = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
        blank_image = PIL.Image.new('RGBA', (b, a), (0, 0, 0, 0))
        blank_image.paste(img_pil, None, mask_pil)
        if type(label) == dict:
            label = json.dumps(label).replace('/','_')
        label_dir = osp.join(out_dir, label)
        out_file = '{}_{}_{}.png'.format(label, json_file, str(i).zfill(zpad))
        shape_bbox = np.array((np.min(points,0), np.max(points,0))).astype(np.uint32).flatten()
        out_img = blank_image.crop(shape_bbox)
        if min(out_img.size) == 0:
            continue
        if not osp.exists(label_dir):
            os.mkdir(label_dir)
        out_img.save(osp.join(label_dir, out_file))
        #print('---')
    print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
