"""Cohen-Sutherland Algorithm"""


INSIDE  = 0b0000
LEFT    = 0b0001
RIGHT   = 0b0010
TOP     = 0b0100
BOTTOM  = 0b1000


def compute_outcode(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:
        code |= LEFT
    elif x > x_max:
        code |= RIGHT
    if y < y_min:
        code |= TOP
    elif y > y_max:
        code |= BOTTOM
    return code


def clip_line(x1: float, y1: float, x2: float, y2: float):
    """Cohen-Sutherland clipping algorithm clips a line from
    P1 = (x1, y1) to P2 = (x2, y2) against a rectangle with 
    diagonal from (x_min, y_min) to (x_max, y_max).
    """
    x_min, y_min, x_max, y_max = 0, 0, 128, 128
    code1 = compute_outcode(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_outcode(x2, y2, x_min, y_min, x_max, y_max)
    accept = False
    while True:
        if not (code1 | code2):
            # bitwise OR is 0: both points inside window; trivially accept and exit loop
            accept = True
            break
        elif code1 & code2:
            # bitwise AND is not 0: both points share an outside zone (LEFT, RIGHT, TOP,
			# or BOTTOM), so both must be outside window; exit loop (accept is false)
            break
        else:
            # At least one endpoint is outside the clip rectangle; pick it.
            code = max(code1, code2)
            # No need to worry about divide-by-zero because, in each case, the
            # outcode bit being tested guarantees the denominator is non-zero
            if code & BOTTOM:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code & TOP:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min
            else:
                assert False
            if code == code1:
                x1, y1 = x, y
                code1 = compute_outcode(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2, y2 = x, y
                code2 = compute_outcode(x2, y2, x_min, y_min, x_max, y_max)
    return accept, x1, y1, x2, y2


def clip_lines(lines):
    cliped = []
    for x1, y1, x2, y2 in lines:
        accept, *coords = clip_line(x1, y1, x2, y2)
        if accept:
            cliped.append(coords)
    return cliped
