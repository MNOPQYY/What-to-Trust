import os
import time
import requests
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm
import re
import asyncio
import aiohttp
import aiofiles
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

async def download_image(session, img_url, folder, index, semaphore, downloaded, max_images, lock):
    async with semaphore: 
        async with lock:  
            if downloaded[0] >= max_images: 
                return False

        for attempt in range(3):  
            try:
                async with session.get(img_url, headers=HEADERS, timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")

                    ext = re.findall(r"\.(jpg|jpeg|png|gif|bmp|webp)", img_url.split("?")[0])
                    ext = ext[0] if ext else "jpg"  
                    filename = os.path.join(folder, f"{index}.{ext}")

                    async with aiofiles.open(filename, "wb") as f:
                        await f.write(await response.read())

                    async with lock:  
                        downloaded[0] += 1

                    return True  
            except Exception as e:
                print(f"Failed (Have Tried {attempt+1}/3): {img_url}, error: {e}")

    return False

async def fetch_image_urls(session, keyword, max_images):
    query = quote(keyword)
    base_url = f"https://www.bing.com/images/search?q={query}"
    page = 0
    img_urls = []

    while len(img_urls) < max_images:
        url = f"{base_url}&first={page * 50}&count=50"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Can not visit Bing: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        for img_tag in soup.find_all("a", class_="iusc"):
            try:
                m = re.search(r'"murl":"(.*?)"', img_tag["m"])
                if m:
                    img_urls.append(m.group(1))
                    if len(img_urls) >= max_images:  
                        break
            except KeyError:
                continue

        if len(img_urls) < max_images:
            page += 1  
        else:
            break

    return img_urls

async def fetch_images(session, path_dir, keyword, obj, state, max_images):
    folder = os.path.join(path_dir, obj, state)
    os.makedirs(folder, exist_ok=True)
    
    img_urls = await fetch_image_urls(session, keyword, max_images)
    
    downloaded = [0]  
    semaphore = asyncio.Semaphore(10)  
    lock = asyncio.Lock()  

    tasks = [asyncio.create_task(download_image(session, img_url, folder, index, semaphore, downloaded, max_images, lock)) 
             for index, img_url in enumerate(img_urls, start=1)]

    await asyncio.gather(*tasks)  

async def main(keyword,obj,state, max_images=50):
    async with aiohttp.ClientSession() as session:
        await fetch_images(session, keyword,obj,state, max_images=50)
        
        
if __name__ == "__main__":
    path_dir = 'data/image_knowledge/'
    state_file = json.load(open('data/MOST_state_categories_dict.json','r'))
    for obj in tqdm(state_file,total=len(state_file)):
        state_list = state_file[obj]
        os.makedirs(os.path.join(path_dir,obj), exist_ok=True)
        for state in state_list:
            query = state + ' '+obj
            asyncio.run(main(path_dir, query, obj, state))