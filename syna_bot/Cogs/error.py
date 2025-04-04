# discord api
import discord
from discord.ext import commands

# custom utilities
from syna_bot.Utilities.log.log import Logger
from pathlib import Path
log = Logger("errors")

ROOT_DIR = Path(__file__).resolve().parent.parent

class Errors(commands.Cog):
    def __init__(self, client):
        self.client = client

    @commands.Cog.listener()
    async def on_ready(self):
        await log.info("Errors cog loaded.")

    @commands.Cog.listener()
    async def on_command_error(self, context, error):

        if isinstance(error, commands.CheckFailure):
            await context.reply(
                "You are not priveleged enough to use this command.", 
                mention_author=False
            )

        else:
            await context.reply(
                f"**Error**\n```diff\n- {error}```",
                mention_author=False
            )

def setup(client):
    client.add_cog(Errors(client))