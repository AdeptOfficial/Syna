# discord api
import discord 
from discord.ext import commands

# system
import psutil
import os

# model
from Models.model import Model

# custom utilities and setup
from syna_bot.Utilities.log import log
from syna_bot.Utilities.db import db

log = log.Logger("client")

class Client(commands.AutoShardedBot):
    def __init__(self, *args, model_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # client version
        self.Version = "0.1.0"

        # operational level variables
        self.Updating = False
        self.Debugging = False
        self.Maintaining = False
        self.command_prefix = commands.when_mentioned_or(".")

        # psutil utilization
        self.process = psutil.Process(os.getpid())

        # model path
        self.model = Model(model_path, use_unsloth=True)


        # Load cogs from all subfolders
        cogs_dir = os.path.join(os.path.dirname(__file__), "Cogs")
        for root, _, files in os.walk(cogs_dir):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    relative_path = os.path.relpath(root, os.path.dirname(__file__))
                    module_name = f"syna_bot.{relative_path.replace(os.sep, '.')}.{file[:-3]}"
                    self.load_extension(module_name)

    @commands.Cog.listener()
    async def on_ready(self):
        await log.info(f"{self.user} is online.")
        await log.info(f"Loading fine-tuned model from: {self.model.name}")
        await db.build()
        await log.info("Database built.")
        await log.info(f"Version: {self.Version}")
        await log.info("Bot is ready.")