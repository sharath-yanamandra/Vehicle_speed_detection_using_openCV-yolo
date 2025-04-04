import os
import pandas as pd
from datetime import datetime
import numpy as np
import torch
from time import time
import cv2
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.saved_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)
        self.frame_skip = 2
        
        # Display settings
        self.display_duration = 90  # 3 seconds at 30fps
        self.display_tracks = {}  # Track which vehicles to display
        self.display_start_frames = {}  # When to start displaying each vehicle
        
        # Speed calculation parameters
        self.real_world_distance = kwargs.get('real_world_distance', 20)
        self.fps = 30 // self.frame_skip
        
        # Initialize CSV data storage
        self.csv_filename = self.get_daily_filename()
        self.create_csv()

    def get_daily_filename(self):
        """Generate a filename based on the current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"vehicle_count_data_{current_date}.csv"
        return filename

    def create_csv(self):
        """Create the CSV file with proper headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            header = ["Track ID", "Label", "Action", "Speed (km/h)", "Class", "Date", "Time"]
            df = pd.DataFrame(columns=header)
            df.to_csv(self.csv_filename, index=False)
            print(f"CSV file created: {self.csv_filename} with headers.")

    def save_label_to_file(self, track_id, label, action, speed, class_name):
        """Save the label, track_id, action, speed, date, time, and class name to a CSV file."""
        if isinstance(speed, torch.Tensor):
            speed = speed.item()
        elif isinstance(speed, np.ndarray):
            speed = speed.item()

        speed = int(round(speed))
        
        current_time = datetime.now()
        current_date = current_time.date()
        current_time_str = current_time.strftime("%H:%M:%S")

        data = {
            "Track ID": track_id,
            "Label": label,
            "Action": action,
            "Speed (km/h)": speed,
            "Class": class_name,
            "Date": current_date,
            "Time": current_time_str
        }

        df = pd.DataFrame([data])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)

    def calculate_speed(self, track_id, current_position, previous_position, time_difference):
        """Calculate speed with perspective correction and real-world distance calibration."""
        if time_difference <= 0:
            return 0
            
        # Get pixel distance
        pixel_distance = np.sqrt(
            (current_position[0] - previous_position[0]) ** 2 +
            (current_position[1] - previous_position[1]) ** 2
        )
        
        # Convert pixel distance to real-world distance
        region_pixel_distance = np.sqrt(
            (self.region[1][0] - self.region[0][0]) ** 2 +
            (self.region[1][1] - self.region[0][1]) ** 2
        )
        meters_per_pixel = self.real_world_distance / region_pixel_distance
        real_distance = pixel_distance * meters_per_pixel
        
        # Calculate speed in km/h
        speed = (real_distance / time_difference) * 3.6 * 1.2  # Convert m/s to km/h with 20% adjustment
        
        # Apply adaptive smoothing
        if track_id in self.spd:
            diff = abs(speed - self.spd[track_id])
            if diff < 10:
                speed = 0.8 * self.spd[track_id] + 0.2 * speed
            else:
                speed = 0.6 * self.spd[track_id] + 0.4 * speed
            
            # Limit speed change
            max_change = 15
            prev_speed = self.spd[track_id]
            if abs(speed - prev_speed) > max_change:
                if speed > prev_speed:
                    speed = prev_speed + max_change
                else:
                    speed = prev_speed - max_change
        
        return min(max(speed, 0), 150)  # Cap speed between 0 and 150 km/h

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Count objects and update file based on centroid movements."""
        if prev_position is None or track_id in self.counted_ids:
            return

        action = None
        speed = None

        if len(self.region) == 2:  # Handle linear region counting
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    if current_centroid[0] > prev_position[0]:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                        action = "IN"
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                        action = "OUT"
                else:
                    if current_centroid[1] > prev_position[1]:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                        action = "IN"
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                        action = "OUT"
                self.counted_ids.append(track_id)

        if action:
            label = f"{self.names[cls]} ID: {track_id}"
            speed = self.spd.get(track_id, 0)
            self.save_label_to_file(track_id, label, action, speed, self.names[cls])

    def store_classwise_counts(self, cls):
        """Initialize count dictionary for a given class."""
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """Display the counts and actions on the image."""
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
                                f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """Main counting function with speed display only at line crossing."""
        self.im0 = im0
        
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=2)
        self.extract_tracks(im0)
        self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        # Process each tracked object
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)
            
            current_position = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Calculate speed
            if track_id in self.trk_pt and track_id in self.trk_pp:
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = self.calculate_speed(
                        track_id,
                        current_position,
                        self.trk_pp[track_id],
                        time_difference
                    )
                    self.spd[track_id] = speed

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = current_position

            # Check for line crossing
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            if prev_position is not None and track_id not in self.counted_ids:
                line = self.LineString(self.region)
                if line.intersects(self.LineString([prev_position, current_position])):
                    # Vehicle just crossed the line
                    self.display_tracks[track_id] = True
                    self.display_start_frames[track_id] = len(self.track_history[track_id])
                    self.count_objects(current_position, track_id, prev_position, cls)

            # Display speed only for vehicles that have crossed the line
            if track_id in self.display_tracks:
                frames_since_crossing = len(self.track_history[track_id]) - self.display_start_frames[track_id]
                if frames_since_crossing <= self.display_duration:
                    if track_id in self.spd:
                        speed_text = f"{int(self.spd[track_id])} km/h"
                        self.annotator.box_label(
                            box,
                            label=speed_text,
                            color=(255, 0, 0)  # Blue color in BGR format
                        )
                else:
                    # Remove from display after duration
                    self.display_tracks.pop(track_id, None)
                    self.display_start_frames.pop(track_id, None)

        self.display_counts(im0)
        return im0

def main():
    # RTSP stream URL - replace with your RTSP stream URL
    rtsp_url = "rtsp://username:password@ip_address:port/stream"
    
    # Initialize video capture with RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    
    # Add RTSP-specific settings to improve stability
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize buffer size
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))  # Use H264 codec
    
    # Check if stream is opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open RTSP stream. Please check the URL and network connection.")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate line points leaving 10 pixels on each side
    scale_x = frame_width / 1020
    scale_y = frame_height / 500
    left_margin = 10
    right_margin = 10
    line_y = int(350 * scale_y)
    
    region_points = [
        (int(left_margin * scale_x), line_y),
        (int((1020 - right_margin) * scale_x), line_y)
    ]
    
    # Initialize object counter
    counter = ObjectCounter(
        region=region_points,
        model="path/to/your/yolov8x.pt",  # Update with your model path
        classes=[2, 5, 7],
        show_in=True,
        show_out=True,
        line_width=2,
        real_world_distance=100
    )
    
    # Set up video writer for saving the processed stream (optional)
    output_filename = f"rtsp_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps//2, (frame_width, frame_height))
    
    # Variables for frame skipping and reconnection
    count = 0
    max_reconnect_attempts = 5
    reconnect_attempts = 0
    frame_skip = 2  # Process every other frame
    
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from stream")
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnect_attempts:
                        print("Max reconnection attempts reached. Exiting...")
                        break
                    
                    print(f"Attempting to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                    cap.release()
                    time.sleep(2)  # Wait before attempting to reconnect
                    cap = cv2.VideoCapture(rtsp_url)
                    continue
                
                # Reset reconnection counter on successful frame grab
                reconnect_attempts = 0
                
                # Skip frames based on frame_skip value
                count += 1
                if count % frame_skip != 0:
                    continue
                
                # Process the frame
                processed_frame = counter.count(frame)
                
                # Write the processed frame (optional)
                video_writer.write(processed_frame)
                
                # Display the frame
                display_frame = cv2.resize(processed_frame, (1020, 500))
                cv2.imshow("RTSP Stream", display_frame)
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
    
    finally:
        # Release resources
        print("Cleaning up resources...")
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()