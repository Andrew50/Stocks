import discord
import discordwebhook
from Data import Data as data
class discordManager(): 
	def initialize():
		intents = discord.Intents.default()
		intents.message_content = True
		client = discord_bot(intents=intents)
		client.run('MTE2MDAyNzYzMzY1Mjg3NTMyNQ.GDB8fq.ELueCj6j1lGnYE0iLGIHUD51hzWm_JdyqaNrK0')
class discord_bot(discord.Client): 
	async def on_ready(self):
		pass
	async def on_message(self, message):
		message_content = message.content
		message_author = message.author
		if message.content.startswith('query?'):
			await message.channel.send('Query Initiated')
			text = message_content.split('?').split(' ')
			try:
				ticker = text[0]
				date = text[1]
				tf = text[2]
				numBars = text[3]
				df = data.get(ticker, tf, dt=date, bars=numBars)
			except:
				await message.channel.send('Incorrect format used.')
				
if __name__ == '__main__':
	pass

				

