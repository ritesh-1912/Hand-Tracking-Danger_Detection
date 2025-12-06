# Hand Danger Detection POC

Real-time hand tracking that detects when your hand approaches a virtual boundary and triggers warnings.

## Approach

Used classical CV instead of pose detection libraries:

- HSV color segmentation for skin detection
- Contour analysis to find hand position
- Distance-based state machine (SAFE/WARNING/DANGER)

## Implementation

The system converts frames to HSV, filters for skin tones, finds the largest contour (hand), calculates distance to a boundary line, and displays appropriate warnings.

Processing at reduced resolution (320x240) then scaling back for display gives ~40 FPS on CPU.

## Running

```bash
pip3 install opencv-python numpy
python3 handtracking.py
```

Press 'Q' to quit, 'S' for speed mode.

## Performance

Achieves 30-50 FPS on MacBook (exceeds 8 FPS requirement). Works reliably in normal lighting conditions.

## Limitations

- Lighting dependent (struggles in very dim conditions)
- Single hand tracking only
- False positives with other skin-colored objects in frame

## Future Improvements

Add background subtraction, fingertip detection via convex hull, multi-hand support, and audio alerts.
