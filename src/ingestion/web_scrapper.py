import os 
import re
import json
import requests
import warnings
import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup

# Load config file 
config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)


# Scrap list of links
class StoreDevganLinks:
    def __init__(self, url):
        self.url = url 
        self.section_type = url.split('_')[-1].split('.')[0] # APPLIED FOR DEVGAN LAWS LINK 
        print(self.section_type)
    
    def scrape_links(self):
        respone = requests.get(self.url)
        soup = BeautifulSoup(respone.content, 'html.parser')
        
        links = [link.get('href') for link in soup.find_all('a')]
        urls = [f'https://devgan.in{link}' for link in links if link and link.startswith(f'/{self.section_type}/section/')]
        return {"urls": urls}

    def get_urls(self):
        data = self.scrape_links()
        
        with open(f'./data/{self.section_type}_urls.json', 'w') as file:
            json.dump(data, file, indent=2)


# Scrap data from each url
async def scrape_urls(url_data, section_type):
    output_data = []

    async with ClientSession() as session:
        for url in url_data['urls']:
            # Extract the section from the URL
            section = url.split('/')[-2]

            # Make the HTTP request and get the HTML content
            async with session.get(url) as response:
                html_content = await response.text()

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Scrape the desired elements
            p_elements = [p.get_text() for p in soup.find_all('p')]
            h2_elements = [h2.get_text() for h2 in soup.find_all('h2')]
            li_elements = [li.get_text().strip() for li in soup.find_all('li')]
            # a_elements = [a.get_text() for a in soup.find_all('a')]
            # href_elements = [a.get('href', '') for a in soup.find_all('a')]

            # Combine the scraped elements into a single dictionary
            data = {
                'content': '\n'.join(p_elements + h2_elements + li_elements),
                'section': section_type + '-' + section
            }

            # Add the data to the output list
            output_data.append(data)

    return output_data

async def main(section_type):
    # Load the JSON data
    with open(f'./data/{section_type}_urls.json', 'r') as file:
        url_data = json.load(file)

    # Scrape the URLs
    output_data = await scrape_urls(url_data, section_type)

    # Save the output data to a JSON file
    with open(f'./data/{section_type}_data.json', 'w') as file:
        json.dump(output_data, file, indent=2)

if __name__ == '__main__':
    asyncio.run(main("jja"))
    # links = StoreDevganLinks(url=config['DEVGAN_LAW_LINKS']['JJA'])
    # links.scrape_links()
    # links.get_urls()