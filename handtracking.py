import cv2
import numpy as np
import time

class HandTrackingSystem:
    def __init__(self):
        """Initialize the hand tracking system"""
        #color range for skin detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # vertical line in middle
        self.boundary_x = 320
        
        # distance threshold
        self.SAFE_DISTANCE = 150
        self.WARNING_DISTANCE = 50
        
        # FPS calclation
        self.prev_time = 0
        
        self.state_colors = {
            'SAFE': (0, 255, 0),
            'WARNING': (0, 165, 255),
            'DANGER': (0, 0, 255)
        }
    
    def detect_skin(self, frame):
        """Detect skin-colored regions in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def find_hand_position(self, mask):
        """Find the position of the hand from the skin mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # finding  largest contor
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 1000:
            return None
        
        # center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), largest_contour
        
        return None
    
    def calculate_distance(self, hand_pos):
        """Calculate horizontal distance from hand to boundary"""
        if hand_pos is None:
            return None
        return abs(hand_pos[0] - self.boundary_x)
    
    def get_state(self, distance):
        """Determine current state based on distance"""
        if distance is None:
            return 'SAFE'
        
        if distance <= self.WARNING_DISTANCE:
            return 'DANGER'
        elif distance <= self.SAFE_DISTANCE:
            return 'WARNING'
        else:
            return 'SAFE'
    
    def draw_ui(self, frame, hand_data, state, distance, fps):
        """Draw all UI elements on the frame"""
        h, w = frame.shape[:2]
        
        # virtual boundary
        cv2.line(frame, (self.boundary_x, 0), (self.boundary_x, h), (255, 255, 255), 3)
        cv2.putText(frame, 'BOUNDARY', (self.boundary_x - 50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # hand position and contour
        if hand_data:
            hand_pos, contour = hand_data
            cv2.circle(frame, hand_pos, 10, (255, 0, 255), -1)
            cv2.drawContours(frame, [contour], 0, (0, 255, 255), 2)
            cv2.line(frame, hand_pos, (self.boundary_x, hand_pos[1]), (255, 255, 0), 2)
        
        # state information box
        state_color = self.state_colors[state]
        cv2.rectangle(frame, (0, 0), (w, 80), state_color, -1)
        cv2.putText(frame, state, (w//2 - 80, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        
        # warning (flash)
        if state == 'DANGER':
            flash = int(time.time() * 4) % 2
            if flash:
                cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, 'DANGER DANGER', (w//2 - 200, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        
        # Distance and FPS info
        dist_text = f'{distance}' if distance else 'N/A'
        info_text = f'Distance: {dist_text} px | FPS: {fps:.1f}'
        cv2.putText(frame, info_text, (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, 'Move hand towards the white line', (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'Press Q to quit', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop to run the hand tracking system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.boundary_x = w // 2
        
        print("Starting Hand Tracking System...")
        print("Instructions:")
        print("- Show your hand to the camera")
        print("- Move it towards the white boundary line")
        print("- Watch the state change: SAFE -> WARNING -> DANGER")
        print("- Press 'Q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time
            
            skin_mask = self.detect_skin(frame)
            hand_data = self.find_hand_position(skin_mask)
            
            # hand position
            hand_pos = hand_data[0] if hand_data else None
            
            # Calculate distance and state
            distance = self.calculate_distance(hand_pos)
            state = self.get_state(distance)
            
            # Draw UI
            frame = self.draw_ui(frame, hand_data, state, distance, fps)
            
            # Display frame
            cv2.imshow('Hand Tracking - Danger Detection', frame)
            
            # Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")


if __name__ == "__main__":
    system = HandTrackingSystem()
    system.run()