#!/usr/bin/env python3

import simplejson as json
import math
import numpy as np
import sys
import decimal as d
from decimal import Decimal
from stage import X_RANGE, Y_RANGE

from PIL import Image


def make_image(vals, dest_file):
    width = int(math.sqrt(len(vals)))
    image_vals = np.reshape(vals, (width, width))

    print(image_vals)
    print(len(image_vals))
    print(len(image_vals[0]))

    image = Image.new("1", (width, width))
    data = image.load()

    for i in range(width):
        for j in range(width):
            # print(image_vals[i][j], end="")
            data[i, j] = int(image_vals[i][j])

    x_width = X_RANGE[1] - X_RANGE[0]
    y_width = Y_RANGE[1] - Y_RANGE[0]

    # image = Image.fromarray(image_vals, mode="1")
    image.resize((2000, int(2000 * (y_width / x_width)))).save(dest_file)


def calyx_mandlebrot(src_file="results.json", dest_file="mandelbrot.png"):
    data = None
    with open(src_file) as f:
        data = json.load(f, use_decimal=True)

    vals = data["memories"]["int_outputs0"]
    make_image(vals, dest_file)


def python_mandlebrot(n_iters):
    data = None
    with open("results.json") as f:
        data = json.load(f, use_decimal=True)

    with d.localcontext() as ctx:
        ctx.prec = 32

        c_real_mem = list(map(lambda x: Decimal(x), data["memories"]["int_c_real0"]))
        c_img_mem = list(map(lambda x: Decimal(x), data["memories"]["int_c_img0"]))

        size = len(c_real_mem)

        z_real_mem = [Decimal("0")] * size
        z_img_mem = [Decimal("0")] * size
        outputs = [1] * size

        print(type(z_real_mem[0]))

        for i in range(size):
            c_real = c_real_mem[i]
            c_img = c_img_mem[i]
            for _ in range(int(n_iters)):
                z_real = z_real_mem[i]
                z_img = z_img_mem[i]
                if ((z_img * z_img) + (z_real * z_real)) > 4 or outputs[i] == 0:
                    print(i)
                    outputs[i] = 0
                else:
                    z_real_mem[i] = (z_real * z_real) - (z_img * z_img) + c_real
                    z_img_mem[i] = 2 * (z_img * z_real) + c_img
        make_image(outputs)


def main():
    # if len(sys.argv) > 1:
    #     python_mandlebrot(sys.argv[1])
    # else:
    calyx_mandlebrot(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
