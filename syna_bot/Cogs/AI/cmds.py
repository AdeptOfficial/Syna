import discord
from discord.ext import commands
from discord.commands import slash_command, Option

# custom utilities
from syna_bot.Utilities.log.log import Logger
log = Logger("AI-cog")

class cmds(commands.Cog):
    def __init__(self, client):
        self.client = client


    @commands.Cog.listener()
    async def on_ready(self):
        await log.info("AI cog loaded.")

    @commands.slash_command(
        description="A simple command to say hello."
    )
    async def hello(self, ctx):
        await ctx.respond("Hello!")

    @commands.slash_command(
        description="A command to check the bot's latency."
    )
    async def ping(self, ctx):
        latency = round(self.bot.latency * 1000)  # Convert to milliseconds
        await ctx.respond(f"Pong! Latency is {latency}ms.")

    @commands.slash_command(
        description="Ask the bot a question.",
        options=[
            Option(
                name="question",
                description="The question you want to ask. Response with trained AI model.",
                type=3,
                required=True
            )
        ]
    )
    async def ask(self, ctx, question: str):
        #await ctx.respond(f"You asked: {question}")
        await log.info(f"Question asked: {question}")
        # update to use model to generate response
        await ctx.defer()
        response = await self.client.model.ask(question)
        await ctx.respond(response)
        

def setup(bot):
    bot.add_cog(cmds(bot))