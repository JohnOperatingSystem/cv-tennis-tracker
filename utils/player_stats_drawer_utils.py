import cv2
import math


REFERENCE_FRAME_WIDTH = 1280
REFERENCE_FRAME_HEIGHT = 720


def _format_speed(value):
    if value is None:
        return "--"
    try:
        if not math.isfinite(float(value)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"{float(value):.1f}"


def get_stats_board_layout(frame_shape):
    """Return a uniformly scaled version of the 1280x720 stats layout."""
    if len(frame_shape) < 2:
        raise ValueError("Frame shape must contain height and width")
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be greater than zero")

    scale = min(
        frame_width / REFERENCE_FRAME_WIDTH,
        frame_height / REFERENCE_FRAME_HEIGHT,
    )

    def scaled(value):
        return max(1, int(round(value * scale)))

    width = scaled(350)
    height = scaled(300)
    right_margin = scaled(50)
    mini_court_width = scaled(250)
    panel_gap = scaled(20)
    top_margin = scaled(50)
    start_x = (
        frame_width
        - right_margin
        - mini_court_width
        - panel_gap
        - width
    )
    start_y = top_margin
    return {
        "scale": scale,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": start_x + width,
        "end_y": start_y + height,
        "width": width,
        "height": height,
    }


def draw_player_stats(output_video_frames, player_stats, point_result=None):
    frame_count = min(len(output_video_frames), len(player_stats))

    for frame_index, (_, row) in enumerate(
        player_stats.iloc[:frame_count].iterrows()
    ):
        player_1_shot_speed = _format_speed(row["player_1_last_shot_speed"])
        player_2_shot_speed = _format_speed(row["player_2_last_shot_speed"])
        player_1_speed = _format_speed(row["player_1_last_player_speed"])
        player_2_speed = _format_speed(row["player_2_last_player_speed"])
        ball_speed = _format_speed(row["ball_speed"])

        avg_player_1_shot_speed = _format_speed(
            row["player_1_average_shot_speed"]
        )
        avg_player_2_shot_speed = _format_speed(
            row["player_2_average_shot_speed"]
        )
        avg_player_1_speed = _format_speed(
            row["player_1_average_player_speed"]
        )
        avg_player_2_speed = _format_speed(
            row["player_2_average_player_speed"]
        )

        frame = output_video_frames[frame_index]
        layout = get_stats_board_layout(frame.shape)
        scale = layout["scale"]
        start_x = layout["start_x"]
        start_y = layout["start_y"]

        def scaled(value):
            return max(1, int(round(value * scale)))

        def draw_text(
            text,
            offset_x,
            offset_y,
            font_scale,
            thickness,
            color=(255, 255, 255),
        ):
            rendered_font_scale = max(0.1, font_scale * scale)
            available_width = scaled(350 - offset_x - 8)
            text_width = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                rendered_font_scale,
                max(1, scaled(thickness)),
            )[0][0]
            if text_width > available_width:
                rendered_font_scale = max(
                    0.1,
                    rendered_font_scale * available_width / text_width,
                )
            cv2.putText(
                frame,
                text,
                (start_x + scaled(offset_x), start_y + scaled(offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                rendered_font_scale,
                color,
                max(1, scaled(thickness)),
                cv2.LINE_AA,
            )

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (start_x, start_y),
            (layout["end_x"], layout["end_y"]),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        draw_text("     Player 1     Player 2", 80, 30, 0.6, 2)
        draw_text("Shot Speed", 10, 80, 0.45, 1)
        draw_text(
            f"{player_1_shot_speed} km/h    {player_2_shot_speed} km/h",
            130,
            80,
            0.5,
            2,
        )

        draw_text("Player Speed", 10, 120, 0.45, 1)
        draw_text(
            f"{player_1_speed} km/h    {player_2_speed} km/h",
            130,
            120,
            0.5,
            2,
        )

        draw_text("avg. S. Speed", 10, 160, 0.45, 1)
        draw_text(
            f"{avg_player_1_shot_speed} km/h    "
            f"{avg_player_2_shot_speed} km/h",
            130,
            160,
            0.5,
            2,
        )

        draw_text("avg. P. Speed", 10, 200, 0.45, 1)
        draw_text(
            f"{avg_player_1_speed} km/h    {avg_player_2_speed} km/h",
            130,
            200,
            0.5,
            2,
        )

        draw_text("Ball Speed", 10, 240, 0.45, 1)
        quality = row.get("ball_speed_quality", "unavailable")
        draw_text(
            f"{ball_speed} km/h ({quality})",
            130,
            240,
            0.48,
            1,
        )

        if (
            point_result is not None
            and point_result.terminal_frame is not None
            and frame_index >= point_result.terminal_frame
        ):
            if point_result.winner_id is None:
                result_text = "Result: UNKNOWN"
            else:
                result_text = (
                    f"Winner: P{point_result.winner_id} "
                    f"({point_result.reason.upper()})"
                )
            draw_text(
                result_text,
                10,
                280,
                0.55,
                2,
                color=(0, 255, 255),
            )

    return output_video_frames
