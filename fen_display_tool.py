from PIL import Image, ImageDraw, ImageFont
import chess

# Size settings
SQUARE_SIZE = 60
BOARD_SIZE = SQUARE_SIZE * 8
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)

# Map chess pieces to unicode chess symbols (for simple text drawing)
PIECE_UNICODE = {
    chess.PAWN:   {'white': '\u2659', 'black': '\u265F'},
    chess.ROOK:   {'white': '\u2656', 'black': '\u265C'},
    chess.KNIGHT: {'white': '\u2658', 'black': '\u265E'},
    chess.BISHOP: {'white': '\u2657', 'black': '\u265D'},
    chess.QUEEN:  {'white': '\u2655', 'black': '\u265B'},
    chess.KING:   {'white': '\u2654', 'black': '\u265A'},
}

def draw_chessboard(fen, output_path='chessboard.png'):
    board = chess.Board(fen)
    image = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), LIGHT_COLOR)
    draw = ImageDraw.Draw(image)

    # Load a font supporting chess unicode symbols (default font may work on some systems)
    # You can replace with a path to a TTF font if needed
    font_size = int(SQUARE_SIZE * 0.75)
    font = ImageFont.load_default()
    
    for rank in range(8):
        for file in range(8):
            square_color = LIGHT_COLOR if (rank + file) % 2 == 0 else DARK_COLOR
            x0 = file * SQUARE_SIZE
            y0 = rank * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            draw.rectangle([x0, y0, x1, y1], fill=square_color)

            square_index = chess.square(file, 7 - rank)  # chess module uses 0= a1 bottom-left
            piece = board.piece_at(square_index)
            if piece:
                color_str = 'white' if piece.color else 'black'
                symbol = PIECE_UNICODE[piece.piece_type][color_str]
                text_width, text_height = draw.textsize(symbol, font=font)
                text_x = x0 + (SQUARE_SIZE - text_width) / 2
                text_y = y0 + (SQUARE_SIZE - text_height) / 2 - 5  # Adjust vertically
                draw.text((text_x, text_y), symbol, fill='black' if piece.color else 'white', font=font)
    
    image.save(output_path)
    print(f"Saved chessboard image to {output_path}")

if __name__ == "__main__":
    fen_file = 'detected_fen.txt'
    with open(fen_file, 'r') as file:
        fen = file.read().strip()
    draw_chessboard(fen)
