import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import chess

# Update this path to your finetuned YOLOv8 weights
MODEL_PATH = r'D:\Codes\fdss\finetuned_training_runs\chesspiece_finetune\weights\best.pt'

# Class names must match your training dataset order
CLASS_NAMES = ['bishop', 'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
  'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

# Map class names to FEN notation (upper = white, lower = black)
FEN_MAP = {
  'black-pawn': 'p', 'black-knight': 'n', 'black-bishop': 'b', 'black-rook': 'r', 'black-queen': 'q', 'black-king': 'k',
  'white-pawn': 'P', 'white-knight': 'N', 'white-bishop': 'B', 'white-rook': 'R', 'white-queen': 'Q', 'white-king': 'K'
}

# Chessboard drawing params
SQUARE_SIZE = 60
BOARD_SIZE = SQUARE_SIZE * 8
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)

PIECE_UNICODE = {
    chess.PAWN:   {'white': '\u2659', 'black': '\u265F'},
    chess.ROOK:   {'white': '\u2656', 'black': '\u265C'},
    chess.KNIGHT: {'white': '\u2658', 'black': '\u265E'},
    chess.BISHOP: {'white': '\u2657', 'black': '\u265D'},
    chess.QUEEN:  {'white': '\u2655', 'black': '\u265B'},
    chess.KING:   {'white': '\u2654', 'black': '\u265A'},
}

def generate_fen_from_detections(detections, board_rows=8, board_cols=8):
    # Initialize empty board squares with '1' for empty
    board = [['1' for _ in range(board_cols)] for _ in range(board_rows)]

    # Define chessboard bounding box on the frame, adjust as per your camera setup
    board_x_min, board_y_min, board_x_max, board_y_max = 100, 100, 540, 540
    cell_width = (board_x_max - board_x_min) / board_cols
    cell_height = (board_y_max - board_y_min) / board_rows

    for *box, cls_id, conf in detections:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        
        if not (board_x_min <= x_center <= board_x_max and board_y_min <= y_center <= board_y_max):
            continue
        
        col = int((x_center - board_x_min) / cell_width)
        row = int((y_center - board_y_min) / cell_height)
        
        fen_row = board_rows - 1 - row  # Convert to FEN rank
        
        cls_name = CLASS_NAMES[int(cls_id)]
        fen_char = FEN_MAP.get(cls_name, '')
        if fen_char:
            board[fen_row][col] = fen_char

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
    
    # Basic default extras (white to move, all castling rights, no en passant, 0 halfmove, 1 fullmove)
    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    return fen

def draw_chessboard(fen_str):
    board = chess.Board(fen_str)
    image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), LIGHT_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", int(SQUARE_SIZE * 0.8))


    for rank in range(8):
        for file in range(8):
            color = LIGHT_COLOR if (rank+file) % 2 == 0 else DARK_COLOR
            x0 = file * SQUARE_SIZE
            y0 = rank * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            draw.rectangle([x0, y0, x1, y1], fill=color)
            square_index = chess.square(file, 7 - rank)
            piece = board.piece_at(square_index)
            if piece:
                color_str = 'white' if piece.color else 'black'
                symbol = PIECE_UNICODE[piece.piece_type][color_str]
                bbox = draw.textbbox((0,0), symbol, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                draw.text((x0 + (SQUARE_SIZE - w)/2, y0 + (SQUARE_SIZE - h)/2 - 5), symbol,
                          fill='black' if piece.color else 'white', font=font)
    
    # Display the image instead of saving
    image.show()

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.3)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                detections.append((*box, cls_id, conf))
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.rectangle(frame, (100, 100), (540, 540), (255,0,0), 2)
        fen = generate_fen_from_detections(detections)
        cv2.putText(frame, f"FEN: {fen}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Live Chess Detector", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC quit
            break
        elif key == 32:  # SPACE save FEN and generate board image
            with open("detected_fen.txt", "w") as f:
                f.write(fen)
            print(f"FEN saved: {fen}")
            draw_chessboard(fen)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
