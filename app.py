from datetime import datetime
import os
import re
from syna_bot.bot import Client
import discord  # Import discord module
from discord.ext import commands  # Import commands from discord.ext
# .env configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
TOKEN = os.getenv("BOT_TOKEN")
ADEPT_GUILD_ID = os.getenv("ADEPT_GUILD_ID")
TRACKER_GUILD_ID = os.getenv("TRACKER_GUILD_ID")

LOCAL = True # Set to True to use local files, False to use Hugging Face models

def load_model():
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Get the root directory of the script

    if LOCAL:
        # Use local files
        model_dir = os.path.join(root_dir, "syna-models")  # Set the model directory
        if not os.path.exists(model_dir):
            print(f"Model directory does not exist: {model_dir}")
            exit(1)

        # Ensure the model directory exists and check version number
        model_dirs = [
            os.path.join("syna-models", d) for d in os.listdir("syna-models") 
            if os.path.isdir(os.path.join("syna-models", d)) and "-v" in d
        ]
        if not model_dirs:
            raise FileNotFoundError("No model directories found in 'syna-models'.")
        print(model_dirs)

        # Extract date from directory names
        def extract_date_and_version(dir_name):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", dir_name)
            version_match = re.search(r"-v(\d+)", dir_name)
            date = datetime.strptime(date_match.group(1), "%Y-%m-%d") if date_match else None
            version = int(version_match.group(1)) if version_match else None
            return date, version

        latest_model_dir = max(model_dirs, key=extract_date_and_version)
        model_path = latest_model_dir
        print(f"Latest model directory located: {model_path}")

        print(f"Using local model from path: {model_path}")
    else:
        # Use Hugging Face model
        model_path = "LouizOne/VtuberLLama-3.1-8B" # Replace with the Hugging Face model name
        print(f"Using Hugging Face model: {model_path}")
    return model_path

def main():
    # calling and initializing the client
    client = Client(
        commands.when_mentioned_or("."),
        intents=discord.Intents.all(), 
        case_insensitive=True, 
        help_command=None,
        model_path=load_model(),
    )
    
    client.run(TOKEN, reconnect=True)
    exit(0)


if __name__ == "__main__":
    main()