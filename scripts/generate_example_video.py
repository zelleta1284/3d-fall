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


def draw_scene(frame, bl, br, tr, tl, t):
    h, w, _ = frame.shape

    # Walls and floor
    floor_color = (70, 80, 90)
    wall_color = (110, 120, 130)
    back_wall_color = (120, 130, 140)

    floor = np.array([bl, br, tr, tl], dtype=np.int32)
    cv2.fillPoly(frame, [floor], floor_color)

    # back wall
    wall_top = np.array([[int(w * 0.25), int(h * 0.1)], [int(w * 0.75), int(h * 0.1)], tr.astype(int), tl.astype(int)], dtype=np.int32)
    cv2.fillPoly(frame, [wall_top], back_wall_color)

    # side walls
    left_wall = np.array([[int(w * 0.05), int(h * 0.15)], [int(w * 0.25), int(h * 0.1)], tl.astype(int), bl.astype(int)], dtype=np.int32)
    right_wall = np.array([[int(w * 0.95), int(h * 0.15)], [int(w * 0.75), int(h * 0.1)], tr.astype(int), br.astype(int)], dtype=np.int32)
    cv2.fillPoly(frame, [left_wall], wall_color)
    cv2.fillPoly(frame, [right_wall], wall_color)

    # Floor grid
    for i in range(1, 10):
        v = i / 10.0
        p0 = quad_interp(bl, br, tr, tl, 0, v).astype(int)
        p1 = quad_interp(bl, br, tr, tl, 1, v).astype(int)
        cv2.line(frame, p0, p1, (60, 70, 80), 1)
    for i in range(1, 10):
        u = i / 10.0
        p0 = quad_interp(bl, br, tr, tl, u, 0).astype(int)
        p1 = quad_interp(bl, br, tr, tl, u, 1).astype(int)
        cv2.line(frame, p0, p1, (60, 70, 80), 1)

    # Rug
    rug = quad_from_floor(bl, br, tr, tl, 0.2, 0.2, 0.45, 0.45)
    cv2.fillPoly(frame, [rug], (45, 55, 60))

    # Coffee table
    table = quad_from_floor(bl, br, tr, tl, 0.55, 0.35, 0.75, 0.55)
    cv2.fillPoly(frame, [table], (90, 80, 70))

    # Chair
    chair = quad_from_floor(bl, br, tr, tl, 0.1, 0.5, 0.22, 0.7)
    cv2.fillPoly(frame, [chair], (80, 75, 70))

    # Window on back wall
    window_rect = np.array([
        [int(w * 0.42), int(h * 0.15)],
        [int(w * 0.58), int(h * 0.15)],
        [int(w * 0.6), int(h * 0.32)],
        [int(w * 0.4), int(h * 0.32)],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [window_rect], (170, 180, 200))

    # Glare patch on floor (moves with t)
    glare_center_u = 0.35 + 0.1 * np.sin(t * 2 * np.pi)
    glare_center_v = 0.35 + 0.1 * np.cos(t * 2 * np.pi)
    glare = quad_from_floor(bl, br, tr, tl, glare_center_u - 0.08, glare_center_v - 0.04, glare_center_u + 0.08, glare_center_v + 0.04)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [glare], (200, 200, 210))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic room scan MP4 for SureStep.ai")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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

        draw_scene(frame, bl, br, tr, tl, t)
        writer.write(frame)

    writer.release()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
