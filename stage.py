from fud.stages import SourceType, Stage
import math
import numpy as np
import simplejson as json

X_RANGE = [-2.5, 1]
Y_RANGE = [-1, 1]


def generate_data(conf):
    width = conf["width"]
    precision = conf["precision"]
    # unroll = conf["unroll"]
    # n_iters = conf["n_iters"]
    int_width = conf["int_width"]

    format_j = {
        "numeric_type": "fixed_point",
        "is_signed": True,
        "width": precision,
        "int_width": int_width,
    }
    data = {
        "int_c_real0": {"data": [], "format": format_j},
        "int_c_img0": {"data": [], "format": format_j},
        "int_outputs0": {
            "data": [1] * width * width,
            "format": {"numeric_type": "bitnum", "is_signed": False, "width": 1},
        },
    }

    dx = (X_RANGE[1] - X_RANGE[0]) / width
    dy = (Y_RANGE[1] - Y_RANGE[0]) / width

    c_real = np.zeros(shape=(width, width))
    c_img = np.zeros(shape=(width, width))
    for xi in range(width):
        for yi in range(width):
            c_real[xi][yi] = X_RANGE[0] + xi * dx
            c_img[xi][yi] = Y_RANGE[0] + yi * dy
    data["int_c_real0"]["data"] = list(c_real.flatten())
    data["int_c_img0"]["data"] = list(c_img.flatten())

    return json.dumps(data, indent=2, use_decimal=True)


def generate_dahlia(conf):
    width = conf["width"]
    precision = conf["precision"]
    unroll = conf["unroll"]
    n_iters = conf["n_iters"]
    int_width = conf["int_width"]

    # check to make sure that we have a valid size
    assert (width * width) / unroll == (width * width) // unroll

    size = width * width
    size_bits = int(math.ceil(math.log2(size))) + 1
    iter_bits = int(math.ceil(math.log2(n_iters))) + 1

    fix_type = f"fix<{precision}, {int_width}>"

    underflow_check = " || ".join(
        [
            "outputs[i] == 0",
            # f"z_img_2 < (0.0 as {fix_type})",
            # f"z_real_2 < (0.0 as {fix_type})",
            # f"z_img_2 + z_real_2 < (0.0 as {fix_type})",
            f"z_img_2 + z_real_2 > (4.0 as {fix_type})",
        ]
    )

    prog = f"""
import futil("/home/samthomas/Research/mandelbrot-fpga/prims/fp.futil") {{
    def mult(left: {fix_type}, right: {fix_type}): {fix_type};
    def add(left: {fix_type}, right: {fix_type}): {fix_type};
    def sub(left: {fix_type}, right: {fix_type}): {fix_type};
}}

decl int_c_real: {fix_type}[{size}];
decl int_c_img: {fix_type}[{size}];
decl int_outputs: bit<1>[{size}];

let c_real_mem: {fix_type}[{size} bank {unroll}];
let c_img_mem: {fix_type}[{size} bank {unroll}];
let z_img_mem: {fix_type}[{size} bank {unroll}];
let z_real_mem: {fix_type}[{size} bank {unroll}];
let outputs: bit<1>[{size} bank {unroll}];

view c_real_sh = c_real_mem[_: bank 1];
view c_img_sh = c_img_mem[_: bank 1];

// copy inputs into memory
for (let i: ubit<{size_bits}> = 0..{size}) {{
  c_real_sh[i] := int_c_real[i];
  c_img_sh[i] := int_c_img[i];
}}

// initialize z, outputs
for (let i: ubit<{size_bits}> = 0..{size}) unroll {unroll} {{
  z_img_mem[i] := (0.0 as {fix_type});
  z_real_mem[i] := (0.0 as {fix_type});
  outputs[i] := 1; // by default everything is in the set
}}

---

for (let i: ubit<{size_bits}> = 0..{size}) unroll {unroll} {{
  let c_real = c_real_mem[i];
  let c_img = c_img_mem[i];
  for (let n_iters: ubit<{iter_bits}> = 0..{n_iters}) {{
    let z_real = z_real_mem[i];
    let z_img = z_img_mem[i];
    ---
    // let z_img_2 = mult(z_img, z_img);
    let z_img_2 = z_img * z_img;
    // let z_real_2 = mult(z_real, z_real);
    let z_real_2 = z_real * z_real;
    if ({underflow_check}) {{
      outputs[i] := 0;
    }} else {{
      // let s = sub(z_real_2, z_img_2);
      let s = z_real_2 - z_img_2;
      // let a0 = add(s, c_real);
      let a0 = s + c_real;
      z_real_mem[i] := a0;
      // let ixr = mult(z_img, z_real);
      // let doub_ixr = mult((2.0 as {fix_type}), ixr);
      let ixr = z_img * z_real;
      let doub_ixr = (2.0 as {fix_type}) * ixr;
      // let a1 = add(doub_ixr, c_img);
      let a1 = doub_ixr + c_img;
      z_img_mem[i] := a1;
    }}
  }}
}}

---

view outputs_sh = outputs[_: bank 1];

// copy outputs
for (let i: ubit<{size_bits}> = 0..{size}) {{
  int_outputs[i] := outputs_sh[i];
}}"""
    # with open("mandlebrot.fuse", "w") as f:
    #     f.write(prog)
    return prog


class MandelbrotStage(Stage):
    """
    Stage that invokes the mandelbrot frontend
    """

    def __init__(self, config):
        super().__init__(
            "mandelbrot",
            "dahlia",
            SourceType.Path,
            SourceType.Stream,
            config,
            "Generates a mandelbrot kernel.",
        )
        self.setup()

    def _define_steps(self, input_path):
        @self.step()
        def generate_mandelbrot_kernel(
            input_path: SourceType.Path,
        ) -> SourceType.String:
            """
            Invoke Mandelbrot frontend
            """
            with open(input_path) as f:
                conf = json.load(f)
                return generate_dahlia(conf)

        return generate_mandelbrot_kernel(input_path)


class DataStage(Stage):
    """
    Stage that generates mandelbrot data.
    """

    def __init__(self, config):
        super().__init__(
            "mandelbrot",
            "brot_data",
            SourceType.Path,
            SourceType.Stream,
            config,
            "Generates data for mandelbrot.",
        )
        self.setup()

    def _define_steps(self, input_data):
        @self.step()
        def data(input_path: SourceType.Path) -> SourceType.String:
            """
            Invoke Mandelbrot data generation
            """
            with open(input_path) as f:
                conf = json.load(f)
                return generate_data(conf)

        return data(input_data)


__STAGES__ = [MandelbrotStage, DataStage]
