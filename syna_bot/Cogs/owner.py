# discord api
from discord.ext import commands
from discord.commands import slash_command, permissions, Option

# system
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# custom utilities
from syna_bot.Utilities.log.log import Logger
log = Logger("owner")

ROOT_DIR = Path(__file__).resolve().parent.parent

# grab home guild from '.env' and typecast horribly lol
Home_Guild = os.getenv("ADEPT_GUILD_ID")

class Admin(commands.Cog):
    def __init__(self, client):
        self.client = client

    @commands.Cog.listener()
    async def on_ready(self):
        await log.info("Owner cog loaded.")
    
    # Load
    @commands.slash_command(
        guild_ids=[Home_Guild],
        description="Load an extension | owner-only command"
    )
    @commands.is_owner() 
    async def load(
        self, 
        context,
        extension: Option(
            str,
            "Select the extension to load", 
            required=True,
            choices=[
                "error", "owner", "AI.cmds"
            ]
        )
    ):
        self.client.load_extension(f"Cogs.{extension}")
        await context.respond(
            f"Loaded extension: `{extension}`"
        )
        await log.info(f"Loaded extension: {extension}")

    # Unload
    @commands.slash_command(
        guild_ids=[Home_Guild],
        description="Unload an extension | owner-only command"
    )
    @commands.is_owner() 
    async def unload(
        self, 
        context,
        extension: Option(
            str,
            "Select the extension to unload", 
            required=True,
            choices=[
                "error", "owner", "AI.cmds"
            ]
        )
    ):
        self.client.unload_extension(f"Cogs.{extension}")
        await context.respond(
            f"Unloaded extension: `{extension}`"
        )
        await log.info(f"Unloaded extension: {extension}")

    # Reload
    @commands.slash_command(
        guild_ids=[Home_Guild],
        description="Reload an extension | owner-only command"
    )
    @commands.is_owner() 
    async def reload(
        self, 
        context,
        extension: Option(
            str,
            "Select the extension to reload", 
            required=True,
            choices=[
                "error", "owner", "AI.cmds"
            ]
        )
    ):
        self.client.reload_extension(f"syna_bot.Cogs.{extension}")
        await context.respond(
            f"Reloaded extension: `{extension}`"
        )
        await log.info(f"Reloaded extension: {extension}")

    # Shutdown
    @commands.slash_command(
        guild_ids=[Home_Guild],
        description="Shutdown the client | owner-only command"
    )
    @commands.is_owner()
    async def shutdown(
        self,
        context
    ):
        await context.respond(
            "Shutting down."
        )
        await log.info("Shutting down.")
        await self.client.close()

    # Restart 
    @commands.slash_command(
        guild_ids=[Home_Guild],
        description="Restart the client | owner-only command"
    )
    @commands.is_owner()
    async def restart(
        self,
        context
    ):
        await context.respond(
            "Restarting..."
        )
        await log.info("Restarting...")
        await self.client.close()
        os.execl(sys.executable, sys.executable, *sys.argv)

def setup(client):
    client.add_cog(Admin(client))