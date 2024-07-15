import os 
import re
import json
import requests
import warnings
import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup

# Scrap list of links
def scrape_links(url):
    respone = requests.get(url)
    soup = BeautifulSoup(respone.content, 'html.parser')
    
    links = [link.get('href') for link in soup.find_all('a')]
    urls = [f'https://devgan.in{link}' for link in links if link.startswith('/ipc/section/')]

    return {"urls": urls}


def main():
    url = "https://devgan.in/all_sections_ipc.php"
    data = scrape_links(url)
    
    with open('ipc_urls.json', 'w') as file:
        json.dump(data, file, indent=2)


# Scrap data from each url
async def scrape_urls(url_data):
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
            # a_elements = [a.get_text() for a in soup.find_all('a')]
            # href_elements = [a.get('href', '') for a in soup.find_all('a')]

            # Combine the scraped elements into a single dictionary
            data = {
                'content': '\n'.join(p_elements + h2_elements),
                'section': section
            }

            # Add the data to the output list
            output_data.append(data)

    return output_data

async def main():
    # Load the JSON data
    with open('urls.json', 'r') as file:
        url_data = json.load(file)

    # Scrape the URLs
    output_data = await scrape_urls(url_data)

    # Save the output data to a JSON file
    with open('ipc_data.json', 'w') as file:
        json.dump(output_data, file, indent=2)

if __name__ == '__main__':
    asyncio.run(main())