from typing import List, Tuple


def validate_kernel(
    kernels: List[Tuple[int, int]],
    strides: List[Tuple[int, int]],
    paddings: List[Tuple[int, int]],
    input_shape: Tuple[int, int] = (1000, 68),
) -> bool:
    """
    Validate a sequence of 2D convolution layer parameters by simulating the
    output size layer-by-layer from a given input shape.

    Checks:
    - All three lists have the same length (one entry per conv layer)
    - Each tuple has two integers
    - Kernel sizes and strides are positive; paddings are non-negative
    - Propagates (H, W) using: out = floor((in + 2*p - k) / s) + 1
    - Each intermediate output dimension must be >= 1

    Prints the output shape after each mock layer. Returns True if all layers
    are valid for the provided input shape; otherwise False.
    """

    if not (len(kernels) == len(strides) == len(paddings)):
        return False

    if not (
        isinstance(input_shape, tuple)
        and len(input_shape) == 2
        and all(isinstance(v, int) for v in input_shape)
        and input_shape[0] > 0
        and input_shape[1] > 0
    ):
        return False

    h, w = input_shape

    for idx, (kernel, stride, padding) in enumerate(zip(kernels, strides, paddings)):
        # if not (
        #     isinstance(kernel, tuple)
        #     and isinstance(stride, tuple)
        #     and isinstance(padding, tuple)
        #     and len(kernel) == 2
        #     and len(stride) == 2
        #     and len(padding) == 2
        # ):
        #     return False

        kh, kw = kernel
        sh, sw = stride
        ph, pw = padding

        if not all(isinstance(v, int) for v in (kh, kw, sh, sw, ph, pw)):
            return False

        if kh <= 0 or kw <= 0:
            return False
        if sh <= 0 or sw <= 0:
            return False
        if ph < 0 or pw < 0:
            return False

        # Simulate convolution output sizes per dimension (dilation=1)
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        if h_out <= 0 or w_out <= 0:
            return False

        print(f"After layer {idx + 1}: (H, W) = ({h_out}, {w_out})")

        h, w = h_out, w_out

    return True




if __name__ == "__main__":
    kernels = [tuple(x) for x in [[100, 3], [15, 68], [5, 1]]]
    strides = [tuple(x) for x in [[4, 1], [2, 1], [1, 1]]]
    paddings = [tuple(x) for x in [[0, 1], [0, 0], [0, 0]]]
    input_shape = (1000, 68)
    print(validate_kernel(kernels, strides, paddings, input_shape)) # type: ignore