from urllib.parse import quote
from urllib.parse import unquote  
import requests

filename = quote("received_日本語音源-2.wav")  # 日本語は URL エンコード
url = f"https://8vm9dxp402l5oh-5000.proxy.runpod.net/download/{filename}"

r = requests.get(url)
with open("downloaded_file.wav", "wb") as f:
    f.write(r.content)
