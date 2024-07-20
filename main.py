import discord
from discord.ext import commands
from model import get_class

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'Zalogowaliśmy się jako {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Cześć! Jestem botem, {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            name = attachment.filename
            url = attachment.url
            await attachment.save(name)
            await ctx.send("Your image is saved")
            await ctx.send(get_class(model="keras_model.h5", labels="label.txt", image=name)) 

    else:
        await ctx.send("Nothing here...")  



bot.run("Your TOKEN")