#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def quad_interp(bl, br, tr, tl, u, v):
    # u: 0..1 left->right, v: 0..1 bottom->top (depth)
    bottom = (1 - u) * bl + u * br
    top = (1 - u) * tl + u * tr
    return (1 - v) * bottom + v * top


def quad_from_floor(bl, br, tr, tl, x0, y0, x1, y1):
    # (x,y) in floor coords (0..1), y=depth
    p0 = quad_interp(bl, br, tr, tl, x0, y0)
    p1 = quad_interp(bl, br, tr, tl, x1, y0)
    p2 = quad_interp(bl, br, tr, tl, x1, y1)
    p3 = quad_interp(bl, br, tr, tl, x0, y1)
    return np.array([p0, p1, p2, p3], dtype=np.int32)


def _color_variation(base, scale=12):
    return tuple(int(max(0, min(255, c + np.random.randint(-scale, scale + 1)))) for c in base)


def _draw_shadow(frame, poly, offset=(6, 6), alpha=0.35):
    shadow = poly + np.array(offset, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [shadow], (0, 0, 0))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _fill_floor_boards(frame, bl, br, tr, tl, n_boards=14):
    base = (78, 82, 86)
    for i in range(n_boards):
        v0 = i / n_boards
        v1 = (i + 1) / n_boards
        quad = quad_from_floor(bl, br, tr, tl, 0, v0, 1, v1)
        color = _color_variation(base, scale=10)
        cv2.fillPoly(frame, [quad], color)
        # subtle board seam
        p0 = quad_interp(bl, br, tr, tl, 0, v1).astype(int)
        p1 = quad_interp(bl, br, tr, tl, 1, v1).astype(int)
        cv2.line(frame, p0, p1, (60, 65, 70), 1)


def _draw_rug_pattern(frame, bl, br, tr, tl, x0, y0, x1, y1, color_a, color_b, nx=6, ny=6):
    for i in range(nx):
        for j in range(ny):
            u0 = x0 + (x1 - x0) * (i / nx)
            v0 = y0 + (y1 - y0) * (j / ny)
            u1 = x0 + (x1 - x0) * ((i + 1) / nx)
            v1 = y0 + (y1 - y0) * ((j + 1) / ny)
            quad = quad_from_floor(bl, br, tr, tl, u0, v0, u1, v1)
            color = color_a if (i + j) % 2 == 0 else color_b
            cv2.fillPoly(frame, [quad], color)


def _apply_noise_and_vignette(frame):
    h, w, _ = frame.shape
    noise = np.random.normal(0, 4, (h, w, 1)).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    vignette = 1 - 0.25 * (x * x + y * y)
    vignette = np.clip(vignette, 0.7, 1.0)
    vignette = vignette[:, :, None]
    out = (noisy.astype(np.float32) * vignette).astype(np.uint8)
    return out


def draw_scene(frame, bl, br, tr, tl, t, layout):
    h, w, _ = frame.shape

    # Walls and floor
    floor = np.array([bl, br, tr, tl], dtype=np.int32)
    _fill_floor_boards(frame, bl, br, tr, tl)

    # back wall
    wall_top = np.array(
        [
            [int(w * 0.25), int(h * 0.1)],
            [int(w * 0.75), int(h * 0.1)],
            tr.astype(int),
            tl.astype(int),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [wall_top], (120, 128, 138))

    # side walls
    left_wall = np.array(
        [
            [int(w * 0.05), int(h * 0.15)],
            [int(w * 0.25), int(h * 0.1)],
            tl.astype(int),
            bl.astype(int),
        ],
        dtype=np.int32,
    )
    right_wall = np.array(
        [
            [int(w * 0.95), int(h * 0.15)],
            [int(w * 0.75), int(h * 0.1)],
            tr.astype(int),
            br.astype(int),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [left_wall], (112, 122, 132))
    cv2.fillPoly(frame, [right_wall], (112, 122, 132))

    # Baseboards
    cv2.line(frame, tuple(bl.astype(int)), tuple(tl.astype(int)), (85, 90, 95), 3)
    cv2.line(frame, tuple(br.astype(int)), tuple(tr.astype(int)), (85, 90, 95), 3)

    if layout == "living":
        rug = quad_from_floor(bl, br, tr, tl, 0.18, 0.18, 0.5, 0.5)
        _draw_rug_pattern(frame, bl, br, tr, tl, 0.18, 0.18, 0.5, 0.5, (55, 60, 65), (45, 50, 55))

        sofa = quad_from_floor(bl, br, tr, tl, 0.58, 0.55, 0.92, 0.82)
        _draw_shadow(frame, sofa, offset=(8, 6), alpha=0.3)
        cv2.fillPoly(frame, [sofa], (90, 95, 100))

        table = quad_from_floor(bl, br, tr, tl, 0.55, 0.32, 0.78, 0.52)
        _draw_shadow(frame, table, offset=(6, 4), alpha=0.25)
        cv2.fillPoly(frame, [table], (100, 90, 80))

        chair = quad_from_floor(bl, br, tr, tl, 0.08, 0.48, 0.23, 0.68)
        _draw_shadow(frame, chair, offset=(5, 4), alpha=0.25)
        cv2.fillPoly(frame, [chair], (85, 80, 75))

        console = quad_from_floor(bl, br, tr, tl, 0.25, 0.75, 0.45, 0.88)
        cv2.fillPoly(frame, [console], (95, 85, 80))

        # Plant
        plant = quad_from_floor(bl, br, tr, tl, 0.12, 0.78, 0.18, 0.86)
        cv2.fillPoly(frame, [plant], (70, 90, 70))
    elif layout == "bedroom":
        bed = quad_from_floor(bl, br, tr, tl, 0.12, 0.33, 0.6, 0.75)
        _draw_shadow(frame, bed, offset=(8, 6), alpha=0.3)
        cv2.fillPoly(frame, [bed], (90, 95, 105))

        quilt = quad_from_floor(bl, br, tr, tl, 0.15, 0.38, 0.52, 0.68)
        cv2.fillPoly(frame, [quilt], (70, 85, 95))

        nightstand = quad_from_floor(bl, br, tr, tl, 0.62, 0.42, 0.76, 0.55)
        _draw_shadow(frame, nightstand, offset=(4, 3), alpha=0.25)
        cv2.fillPoly(frame, [nightstand], (80, 75, 70))

        rug = quad_from_floor(bl, br, tr, tl, 0.05, 0.2, 0.22, 0.32)
        _draw_rug_pattern(frame, bl, br, tr, tl, 0.05, 0.2, 0.22, 0.32, (60, 65, 70), (50, 55, 60), nx=4, ny=3)
    elif layout == "bathroom":
        tub = quad_from_floor(bl, br, tr, tl, 0.08, 0.4, 0.4, 0.75)
        _draw_shadow(frame, tub, offset=(7, 5), alpha=0.3)
        cv2.fillPoly(frame, [tub], (180, 185, 190))

        sink = quad_from_floor(bl, br, tr, tl, 0.55, 0.45, 0.75, 0.6)
        _draw_shadow(frame, sink, offset=(5, 4), alpha=0.25)
        cv2.fillPoly(frame, [sink], (170, 175, 180))

        mat = quad_from_floor(bl, br, tr, tl, 0.2, 0.22, 0.35, 0.32)
        _draw_rug_pattern(frame, bl, br, tr, tl, 0.2, 0.22, 0.35, 0.32, (80, 90, 100), (70, 80, 90), nx=3, ny=2)
    elif layout == "kitchen":
        counter = quad_from_floor(bl, br, tr, tl, 0.05, 0.55, 0.42, 0.88)
        _draw_shadow(frame, counter, offset=(6, 5), alpha=0.25)
        cv2.fillPoly(frame, [counter], (105, 100, 95))

        island = quad_from_floor(bl, br, tr, tl, 0.48, 0.35, 0.8, 0.62)
        _draw_shadow(frame, island, offset=(7, 5), alpha=0.3)
        cv2.fillPoly(frame, [island], (110, 105, 100))

        stool = quad_from_floor(bl, br, tr, tl, 0.82, 0.42, 0.9, 0.52)
        _draw_shadow(frame, stool, offset=(4, 3), alpha=0.25)
        cv2.fillPoly(frame, [stool], (80, 80, 80))

    # Window on back wall
    window_rect = np.array([
        [int(w * 0.42), int(h * 0.15)],
        [int(w * 0.58), int(h * 0.15)],
        [int(w * 0.6), int(h * 0.32)],
        [int(w * 0.4), int(h * 0.32)],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [window_rect], (170, 180, 200))
    cv2.rectangle(frame, (int(w * 0.46), int(h * 0.18)), (int(w * 0.54), int(h * 0.3)), (200, 210, 220), 2)

    # Glare patch on floor (moves with t)
    glare_center_u = 0.35 + 0.08 * np.sin(t * 2 * np.pi)
    glare_center_v = 0.35 + 0.06 * np.cos(t * 2 * np.pi)
    glare = quad_from_floor(bl, br, tr, tl, glare_center_u - 0.08, glare_center_v - 0.04, glare_center_u + 0.08, glare_center_v + 0.04)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [glare], (200, 200, 210))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Subtle wall art for realism
    cv2.rectangle(frame, (int(w * 0.18), int(h * 0.18)), (int(w * 0.3), int(h * 0.28)), (90, 100, 110), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic room scan MP4 for SureStep.ai")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--layout", choices=["living", "bedroom", "kitchen", "bathroom"], default="living")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Ensure mp4v codec is available.")

    for i in range(args.frames):
        t = i / max(args.frames - 1, 1)
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        # Camera sway
        sway = int(30 * np.sin(t * 2 * np.pi))
        depth = int(10 * np.cos(t * 2 * np.pi))

        bl = np.array([int(args.width * 0.1 + sway), int(args.height * 0.95)])
        br = np.array([int(args.width * 0.9 + sway), int(args.height * 0.95)])
        tl = np.array([int(args.width * 0.35 - depth), int(args.height * 0.45)])
        tr = np.array([int(args.width * 0.65 - depth), int(args.height * 0.45)])

        draw_scene(frame, bl, br, tr, tl, t, args.layout)
        frame = _apply_noise_and_vignette(frame)
        writer.write(frame)

    writer.release()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
