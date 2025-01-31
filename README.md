# Telegram Sudoku Solver

This is a Telegram bot that can solve Sudoku puzzles from images using computer vision and deep learning techniques.

## Features
- Accepts an image of an unsolved Sudoku puzzle.
- Uses OpenCV and a trained deep learning model to extract digits from the Sudoku grid.
- Solves the puzzle using Norvig's algorithm.
- Sends back the solved puzzle as an image.

## Installation

### Prerequisites
- Python 3.x
- Telegram Bot API Token (Get one from [BotFather](https://t.me/BotFather))

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Create a `.env` file
Create a `.env` file in the project root and add your Telegram bot token:
```env
API_TOKEN=your_telegram_bot_token_here
```

## Running the Bot
```bash
python bot.py
```

## How It Works
1. The user sends an image of an unsolved Sudoku puzzle.
2. The bot extracts and processes the Sudoku grid using OpenCV.
3. The bot identifies the digits using a trained deep learning model.
4. The extracted digits are fed into Norvig's solver.
5. The solved puzzle is overlaid on the original image and sent back to the user.

## Dependencies
- `python-telegram-bot`
- `opencv-python`
- `numpy`
- `tensorflow`
- `dotenv`

## Contributing
Feel free to fork this repository and submit pull requests to improve the project.

## License
This project is open-source and available under the MIT License.

