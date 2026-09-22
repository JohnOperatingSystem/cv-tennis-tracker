import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_center_of_bbox,
    get_foot_position,
)

class MiniCourt():
    REFERENCE_FRAME_WIDTH = 1280
    REFERENCE_FRAME_HEIGHT = 720

    def __init__(self, frame):
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be greater than zero")
        self.overlay_scale = min(
            frame_width / self.REFERENCE_FRAME_WIDTH,
            frame_height / self.REFERENCE_FRAME_HEIGHT,
        )
        # Keep the court as a compact top-right overlay.  The panel remains
        # tall enough to preserve a regulation court's proportions.
        self.drawing_rectangle_width = self._scaled(165)
        self.drawing_rectangle_height = self._scaled(330)
        self.buffer = self._scaled(24)
        self.padding_court = self._scaled(14)
        self.line_thickness = max(1, self._scaled(2))
        self.point_radius = max(2, self._scaled(5))
        
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def _scaled(self, value):
        return max(1, int(round(value * self.overlay_scale)))
        
    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points = drawing_key_points
        
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
    
    def convert_pixels_to_meters(self, pixels):
        return convert_pixel_distance_to_meters(pixels, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
    
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (2,3)
        ]
        
    def set_canvas_background_box_position(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(
                frame,
                (x, y),
                self.point_radius,
                (255, 0, 0),
                -1,
            )
        
        # draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(
                frame,
                start_point,
                end_point,
                (0, 0, 0),
                self.line_thickness,
            )
        # draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1]+self.drawing_key_points[5])/2))
        cv2.line(
            frame,
            net_start_point,
            net_end_point,
            (255, 0, 0),
            self.line_thickness,
        )
        return frame 
    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # draw rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
    
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_court_homography(self, original_court_key_points):
        if len(original_court_key_points) < 8:
            raise ValueError("At least four court keypoints are required")

        corner_indices = (0, 1, 2, 3)
        source_points = np.float32([
            (
                original_court_key_points[index * 2],
                original_court_key_points[index * 2 + 1],
            )
            for index in corner_indices
        ])
        destination_points = np.float32([
            (
                self.drawing_key_points[index * 2],
                self.drawing_key_points[index * 2 + 1],
            )
            for index in corner_indices
        ])

        if not np.isfinite(source_points).all():
            raise ValueError("Court keypoints contain non-finite coordinates")

        # Polygon order must go around the court rather than crossing it.
        source_polygon = source_points[[0, 1, 3, 2]]
        if abs(cv2.contourArea(source_polygon)) < 1.0:
            raise ValueError("Court corner keypoints do not form a valid area")

        homography = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )
        if not np.isfinite(homography).all():
            raise ValueError("Could not calculate a valid court homography")
        return homography

    def project_point_to_mini_court(self, point, homography):
        point_array = np.float32(point).reshape(1, 1, 2)
        projected_point = cv2.perspectiveTransform(
            point_array,
            homography,
        )[0, 0]
        return float(projected_point[0]), float(projected_point[1])

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, court_keypoints_per_frame):
        if len(player_boxes) != len(court_keypoints_per_frame):
            raise ValueError(
                "Each player-detection frame must have corresponding court keypoints"
            )
        if len(ball_boxes) != len(court_keypoints_per_frame):
            raise ValueError(
                "Each ball-detection frame must have corresponding court keypoints"
            )

        output_player_boxes = []
        output_ball_boxes = []
        
        for frame_num, player_bbox in enumerate(player_boxes):
            original_court_key_points = court_keypoints_per_frame[frame_num]
            homography = self.get_court_homography(original_court_key_points)

            output_player_bboxes_dict = {}
            output_ball_bboxes_dict = {}

            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                output_player_bboxes_dict[player_id] = (
                    self.project_point_to_mini_court(
                        foot_position,
                        homography,
                    )
                )

            ball_box = ball_boxes[frame_num].get(1)
            if ball_box:
                ball_position = get_center_of_bbox(ball_box)
                output_ball_bboxes_dict[1] = (
                    self.project_point_to_mini_court(
                        ball_position,
                        homography,
                    )
                )

            output_player_boxes.append(output_player_bboxes_dict)
            output_ball_boxes.append(output_ball_bboxes_dict)
        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(
                    frame,
                    (x, y),
                    self.point_radius,
                    color,
                    -1,
                )
        return frames
