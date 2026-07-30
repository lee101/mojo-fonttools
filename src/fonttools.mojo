from std.math import sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def read_u16(data: BPtr, pos: Int) -> Int:
    return (Int(data[pos]) << 8) | Int(data[pos + 1])


def qpoint(a: Float64, b: Float64, c: Float64, t: Float64) -> Float64:
    var u = 1.0 - t
    return u * u * a + 2.0 * u * t * b + t * t * c


def cpoint(a: Float64, b: Float64, c: Float64, d: Float64, t: Float64) -> Float64:
    var u = 1.0 - t
    return u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d


def include_point(dst: FPtr, x: Float64, y: Float64):
    if x < dst[0]:
        dst[0] = x
    if y < dst[1]:
        dst[1] = y
    if x > dst[2]:
        dst[2] = x
    if y > dst[3]:
        dst[3] = y


@export("mft_quadratic_point")
def mft_quadratic_point(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, t: Float64, dst_addr: Int
) abi("C"):
    var dst = fp(dst_addr)
    dst[0] = qpoint(x0, x1, x2, t)
    dst[1] = qpoint(y0, y1, y2, t)


@export("mft_cubic_point")
def mft_cubic_point(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, x3: Float64, y3: Float64,
    t: Float64, dst_addr: Int
) abi("C"):
    var dst = fp(dst_addr)
    dst[0] = cpoint(x0, x1, x2, x3, t)
    dst[1] = cpoint(y0, y1, y2, y3, t)


@export("mft_quadratic_bounds")
def mft_quadratic_bounds(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, dst_addr: Int
) abi("C"):
    var dst = fp(dst_addr)
    dst[0] = min(x0, x2)
    dst[1] = min(y0, y2)
    dst[2] = max(x0, x2)
    dst[3] = max(y0, y2)
    var den = x0 - 2.0 * x1 + x2
    if den != 0.0:
        var t = (x0 - x1) / den
        if t >= 0.0 and t < 1.0:
            include_point(dst, qpoint(x0, x1, x2, t), qpoint(y0, y1, y2, t))
    den = y0 - 2.0 * y1 + y2
    if den != 0.0:
        var t = (y0 - y1) / den
        if t >= 0.0 and t < 1.0:
            include_point(dst, qpoint(x0, x1, x2, t), qpoint(y0, y1, y2, t))


def include_cubic_roots(
    dst: FPtr, a: Float64, b: Float64, c: Float64,
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, x3: Float64, y3: Float64
):
    if a == 0.0:
        if b != 0.0:
            var t = -c / b
            if t >= 0.0 and t < 1.0:
                include_point(dst, cpoint(x0, x1, x2, x3, t), cpoint(y0, y1, y2, y3, t))
        return
    var disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return
    var root = sqrt(disc)
    var t = (-b + root) / (2.0 * a)
    if t >= 0.0 and t < 1.0:
        include_point(dst, cpoint(x0, x1, x2, x3, t), cpoint(y0, y1, y2, y3, t))
    t = (-b - root) / (2.0 * a)
    if t >= 0.0 and t < 1.0:
        include_point(dst, cpoint(x0, x1, x2, x3, t), cpoint(y0, y1, y2, y3, t))


@export("mft_cubic_bounds")
def mft_cubic_bounds(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, x3: Float64, y3: Float64,
    dst_addr: Int
) abi("C"):
    var dst = fp(dst_addr)
    dst[0] = min(x0, x3)
    dst[1] = min(y0, y3)
    dst[2] = max(x0, x3)
    dst[3] = max(y0, y3)
    include_cubic_roots(
        dst, 3.0 * (-x0 + 3.0 * x1 - 3.0 * x2 + x3),
        2.0 * (3.0 * x0 - 6.0 * x1 + 3.0 * x2),
        -3.0 * x0 + 3.0 * x1,
        x0, y0, x1, y1, x2, y2, x3, y3
    )
    include_cubic_roots(
        dst, 3.0 * (-y0 + 3.0 * y1 - 3.0 * y2 + y3),
        2.0 * (3.0 * y0 - 6.0 * y1 + 3.0 * y2),
        -3.0 * y0 + 3.0 * y1,
        x0, y0, x1, y1, x2, y2, x3, y3
    )


@export("mft_quadratic_bounds_batch")
def mft_quadratic_bounds_batch(src_addr: Int, count: Int, dst_addr: Int) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    for i in range(count):
        var base = i * 6
        var target = i * 4
        var x0 = src[base]
        var y0 = src[base + 1]
        var x1 = src[base + 2]
        var y1 = src[base + 3]
        var x2 = src[base + 4]
        var y2 = src[base + 5]
        dst[target] = min(x0, x2)
        dst[target + 1] = min(y0, y2)
        dst[target + 2] = max(x0, x2)
        dst[target + 3] = max(y0, y2)
        var den = x0 - 2.0 * x1 + x2
        if den != 0.0:
            var t = (x0 - x1) / den
            if t >= 0.0 and t < 1.0:
                include_point(
                    FPtr(unsafe_from_address=dst_addr + target * 8),
                    qpoint(x0, x1, x2, t), qpoint(y0, y1, y2, t)
                )
        den = y0 - 2.0 * y1 + y2
        if den != 0.0:
            var t = (y0 - y1) / den
            if t >= 0.0 and t < 1.0:
                include_point(
                    FPtr(unsafe_from_address=dst_addr + target * 8),
                    qpoint(x0, x1, x2, t), qpoint(y0, y1, y2, t)
                )


@export("mft_cubic_bounds_batch")
def mft_cubic_bounds_batch(src_addr: Int, count: Int, dst_addr: Int) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    for i in range(count):
        var base = i * 8
        var target = i * 4
        var x0 = src[base]
        var y0 = src[base + 1]
        var x1 = src[base + 2]
        var y1 = src[base + 3]
        var x2 = src[base + 4]
        var y2 = src[base + 5]
        var x3 = src[base + 6]
        var y3 = src[base + 7]
        var target_ptr = FPtr(unsafe_from_address=dst_addr + target * 8)
        target_ptr[0] = min(x0, x3)
        target_ptr[1] = min(y0, y3)
        target_ptr[2] = max(x0, x3)
        target_ptr[3] = max(y0, y3)
        include_cubic_roots(
            target_ptr, 3.0 * (-x0 + 3.0*x1 - 3.0*x2 + x3),
            2.0 * (3.0*x0 - 6.0*x1 + 3.0*x2), -3.0*x0 + 3.0*x1,
            x0, y0, x1, y1, x2, y2, x3, y3
        )
        include_cubic_roots(
            target_ptr, 3.0 * (-y0 + 3.0*y1 - 3.0*y2 + y3),
            2.0 * (3.0*y0 - 6.0*y1 + 3.0*y2), -3.0*y0 + 3.0*y1,
            x0, y0, x1, y1, x2, y2, x3, y3
        )


def length(x: Float64, y: Float64) -> Float64:
    return sqrt(x * x + y * y)


@export("mft_quadratic_arc_length")
def mft_quadratic_arc_length(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64
) abi("C") -> Float64:
    var v0x = -0.492943519233745 * x0 + 0.430331482911935 * x1 + 0.0626120363218102 * x2
    var v0y = -0.492943519233745 * y0 + 0.430331482911935 * y1 + 0.0626120363218102 * y2
    var v1 = length(x2 - x0, y2 - y0) * 0.4444444444444444
    var v2x = -0.0626120363218102 * x0 - 0.430331482911935 * x1 + 0.492943519233745 * x2
    var v2y = -0.0626120363218102 * y0 - 0.430331482911935 * y1 + 0.492943519233745 * y2
    return length(v0x, v0y) + v1 + length(v2x, v2y)


@export("mft_cubic_arc_length")
def mft_cubic_arc_length(
    x0: Float64, y0: Float64, x1: Float64, y1: Float64,
    x2: Float64, y2: Float64, x3: Float64, y3: Float64
) abi("C") -> Float64:
    var total = length(x1 - x0, y1 - y0) * 0.15
    total += length(
        -0.558983582205757*x0 + 0.325650248872424*x1 + 0.208983582205757*x2 + 0.024349751127576*x3,
        -0.558983582205757*y0 + 0.325650248872424*y1 + 0.208983582205757*y2 + 0.024349751127576*y3
    )
    total += length(x3 - x0 + x2 - x1, y3 - y0 + y2 - y1) * 0.26666666666666666
    total += length(
        -0.024349751127576*x0 - 0.208983582205757*x1 - 0.325650248872424*x2 + 0.558983582205757*x3,
        -0.024349751127576*y0 - 0.208983582205757*y1 - 0.325650248872424*y2 + 0.558983582205757*y3
    )
    total += length(x3 - x2, y3 - y2) * 0.15
    return total


@export("mft_decode_simple_glyph")
def mft_decode_simple_glyph(
    src_addr: Int, size: Int, contours: Int, point_count: Int,
    x_addr: Int, y_addr: Int, flags_addr: Int
) abi("C") -> Int:
    var src = bp(src_addr)
    var xs = fp(x_addr)
    var ys = fp(y_addr)
    var flags = bp(flags_addr)
    var pos = 10 + 2 * contours
    if pos + 2 > size:
        return -1
    var instruction_len = read_u16(src, pos)
    pos += 2 + instruction_len
    if pos > size:
        return -1
    var i = 0
    while i < point_count:
        if pos >= size:
            return -1
        var flag = src[pos]
        pos += 1
        flags[i] = flag
        i += 1
        if (Int(flag) & 8) != 0:
            if pos >= size:
                return -1
            var repeats = Int(src[pos])
            pos += 1
            if i + repeats > point_count:
                return -1
            for _ in range(repeats):
                flags[i] = flag
                i += 1
    var value = 0
    for j in range(point_count):
        var flag = Int(flags[j])
        if (flag & 2) != 0:
            if pos >= size:
                return -1
            var delta = Int(src[pos])
            pos += 1
            value += delta if (flag & 16) != 0 else -delta
        elif (flag & 16) == 0:
            if pos + 2 > size:
                return -1
            var raw = read_u16(src, pos)
            pos += 2
            value += raw if raw < 32768 else raw - 65536
        xs[j] = Float64(value)
    value = 0
    for j in range(point_count):
        var flag = Int(flags[j])
        if (flag & 4) != 0:
            if pos >= size:
                return -1
            var delta = Int(src[pos])
            pos += 1
            value += delta if (flag & 32) != 0 else -delta
        elif (flag & 32) == 0:
            if pos + 2 > size:
                return -1
            var raw = read_u16(src, pos)
            pos += 2
            value += raw if raw < 32768 else raw - 65536
        ys[j] = Float64(value)
    return pos


def interpolate_axis(
    coord: Float64, c1: Float64, d1: Float64, c2: Float64, d2: Float64
) -> Float64:
    if c1 > c2:
        return interpolate_axis(coord, c2, d2, c1, d1)
    if c1 == c2:
        return d1 if d1 == d2 else 0.0
    if coord <= c1:
        return d1
    if coord >= c2:
        return d2
    return d1 + (coord - c1) * (d2 - d1) / (c2 - c1)


def interpolate_axis_range(
    coords: FPtr, dst: FPtr, begin: Int, end: Int,
    c1_arg: Float64, d1_arg: Float64, c2_arg: Float64, d2_arg: Float64
):
    var c1 = c1_arg
    var d1 = d1_arg
    var c2 = c2_arg
    var d2 = d2_arg
    if c1 > c2:
        c1 = c2_arg
        d1 = d2_arg
        c2 = c1_arg
        d2 = d1_arg

    comptime W = simd_width_of[DType.float64]()
    var i = begin
    if c1 == c2:
        var value = d1 if d1 == d2 else 0.0
        var values = SIMD[DType.float64, W](value)
        while i + W <= end:
            dst.store(i, values)
            i += W
        while i < end:
            dst[i] = value
            i += 1
        return

    var slope = (d2 - d1) / (c2 - c1)
    var low = SIMD[DType.float64, W](d1)
    var high = SIMD[DType.float64, W](d2)
    while i + W <= end:
        var values = (coords + i * 2).strided_load[width=W](2)
        var interpolated = d1 + (values - c1) * slope
        interpolated = values.le(c1).select(low, interpolated)
        interpolated = values.ge(c2).select(high, interpolated)
        dst.store(i, interpolated)
        i += W
    while i < end:
        dst[i] = interpolate_axis(coords[i * 2], c1, d1, c2, d2)
        i += 1


@export("mft_iup_contour")
def mft_iup_contour(
    coord_addr: Int, point_count: Int, ref_index_addr: Int, ref_delta_addr: Int,
    ref_count: Int, dst_addr: Int
) abi("C"):
    var coords = fp(coord_addr)
    var ox = fp(dst_addr)
    var oy = FPtr(unsafe_from_address=dst_addr + point_count * 8)
    comptime W = simd_width_of[DType.float64]()
    if ref_count == 0:
        var zeros = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= point_count:
            ox.store(i, zeros)
            oy.store(i, zeros)
            i += W
        while i < point_count:
            ox[i] = 0.0
            oy[i] = 0.0
            i += 1
        return
    var ref_indices = ip(ref_index_addr)
    var ref_deltas = fp(ref_delta_addr)
    if ref_count == 1:
        var x_values = SIMD[DType.float64, W](ref_deltas[0])
        var y_values = SIMD[DType.float64, W](ref_deltas[1])
        var i = 0
        while i + W <= point_count:
            ox.store(i, x_values)
            oy.store(i, y_values)
            i += W
        while i < point_count:
            ox[i] = ref_deltas[0]
            oy[i] = ref_deltas[1]
            i += 1
        return

    @parameter
    def process_segment(segment: Int):
        var previous = ref_indices[segment]
        var current_slot = segment + 1
        if current_slot == ref_count:
            current_slot = 0
        var current = ref_indices[current_slot]
        var previous_delta = segment * 2
        var current_delta = current_slot * 2

        ox[previous] = ref_deltas[previous_delta]
        oy[previous] = ref_deltas[previous_delta + 1]
        var c1x = coords[previous * 2]
        var c1y = coords[previous * 2 + 1]
        var c2x = coords[current * 2]
        var c2y = coords[current * 2 + 1]
        var d1x = ref_deltas[previous_delta]
        var d1y = ref_deltas[previous_delta + 1]
        var d2x = ref_deltas[current_delta]
        var d2y = ref_deltas[current_delta + 1]
        if current > previous:
            interpolate_axis_range(coords, ox, previous + 1, current, c1x, d1x, c2x, d2x)
            interpolate_axis_range(coords + 1, oy, previous + 1, current, c1y, d1y, c2y, d2y)
        else:
            interpolate_axis_range(coords, ox, previous + 1, point_count, c1x, d1x, c2x, d2x)
            interpolate_axis_range(coords + 1, oy, previous + 1, point_count, c1y, d1y, c2y, d2y)
            interpolate_axis_range(coords, ox, 0, current, c1x, d1x, c2x, d2x)
            interpolate_axis_range(coords + 1, oy, 0, current, c1y, d1y, c2y, d2y)

    for segment in range(ref_count):
        process_segment(segment)
