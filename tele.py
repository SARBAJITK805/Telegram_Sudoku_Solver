import logging
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from norvig import solve
import norvig
import numpy as np
import imutils
import cv2
import pickle
import os
from nmain import solve_sudoku
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Start command
async def start(update: Update, context):
    await update.message.reply_text("Send me an image of an unsolved Sudoku puzzle, and I'll solve it for you!")

# Handle received images
async def handle_image(update: Update, context):
    try:
        if update.message and update.message.photo:
            logger.info("Image received")
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            input_path = "unsolved_puzzle.jpg"
            await file.download_to_drive(input_path)

            # Solve the Sudoku puzzle
            solved_image_path = solve_sudoku(input_path)

            # Send back the solved image
            with open(solved_image_path, 'rb') as solved_image:
                await update.message.reply_photo(photo=solved_image)

            # Clean up
            os.remove(input_path)
            os.remove(solved_image_path)
        else:
            logger.warning("No photo found in the message")
    except Exception as e:
        logger.error(f"Error handling image: {e}")


# Main function
def main():
    API_TOKEN = os.getenv('API_TOKEN')
    
    # Create the application
    application = Application.builder().token(API_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
