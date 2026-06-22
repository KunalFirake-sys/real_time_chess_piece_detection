import cv2
from ultralytics import YOLO
import numpy as np

# Update your model path here
MODEL_PATH = r'D:\Codes\fdss\finetuned_training_runs\chesspiece_finetune\weights\best.pt'

# Class names in order (must match your dataset)
CLASS_NAMES = [
  'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
  'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

# Helper: Map class names to FEN notation (uppercase=white, lowercase=black)
FEN_MAP = {
  'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b', 'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k',
  'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B', 'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K'
}

def generate_fen_from_detections(detections, board_rows=8, board_cols=8):
    # Create empty board 8x8 with empty spaces
    board = [['1' for _ in range(board_cols)] for _ in range(board_rows)]
    
    # Assume fixed camera and board position; map detections to board squares
    # Define bounding box area for chessboard (adjust according to your camera)
    board_x_min, board_y_min, board_x_max, board_y_max = 100, 100, 540, 540
    cell_width = (board_x_max - board_x_min) / board_cols
    cell_height = (board_y_max - board_y_min) / board_rows

    for *box, cls_id, conf in detections:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        
        # Ignore detections outside board area
        if not (board_x_min <= x_center <= board_x_max and board_y_min <= y_center <= board_y_max):
            continue
        
        col = int((x_center - board_x_min) / cell_width)
        row = int((y_center - board_y_min) / cell_height)
        
        # Board row in chess FEN notation counts from 8 down to 1
        fen_row = board_rows - 1 - row
        
        cls_name = CLASS_NAMES[int(cls_id)]
        fen_char = FEN_MAP.get(cls_name, '')
        if fen_char:
            board[fen_row][col] = fen_char
    
    # Convert each row to FEN string with numbers for empty squares
    fen_rows = []
    for row in board:
        fen_row_str = ''
        empty_count = 0
        for ch in row:
            if ch == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row_str += str(empty_count)
                    empty_count = 0
                fen_row_str += ch
        if empty_count > 0:
            fen_row_str += str(empty_count)
        fen_rows.append(fen_row_str)
    
    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'  # Default FEN extras, you can customize
    return fen

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect chess pieces
        results = model(frame, imgsz=640, conf=0.3)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                detections.append((*box, cls_id, conf))
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}", (x1,y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Draw board outline
        cv2.rectangle(frame, (100,100), (540,540), (255,0,0), 2)

        fen = generate_fen_from_detections(detections)
        cv2.putText(frame, f"FEN: {fen}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Live Chess Detector", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to save FEN to file
            with open('detected_fen.txt', 'w') as f:
                f.write(fen)
            print(f"FEN saved: {fen}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
