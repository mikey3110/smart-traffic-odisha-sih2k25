frame_count = 0
frame_skip_rate = 2 # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process a frame if the counter is a multiple of the skip rate
    if frame_count % frame_skip_rate == 0:
        results = model(frame, classes=[2, 3, 5, 7])
        # ... rest of your code to display and count ...
