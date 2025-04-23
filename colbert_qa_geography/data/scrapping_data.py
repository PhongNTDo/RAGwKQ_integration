import wikipediaapi
from tqdm import tqdm
import os
import time


wiki = wikipediaapi.Wikipedia(user_agent="rag_practicing", language='en')

geo_categories = [
    "Geography",
    "Countries",
    "Cities",
    "Rivers",
    "Mountains",
    "Islands",
    "Lakes",
    "Deserts",
    "Forests",
    "Capital",
    "Sea",
    "Cartography",
    "Urban studies and planning",
    "Demographics",
    "Populated places",
    "Geography by continent",
    "Subdivisions by country",
    "Economy by region"
]


visited_titles = set()
geo_pages = []

def recurse_category(category_page, max_pages=1000000, depth=0):
    global geo_pages
    if depth > 3 or len(geo_pages) > max_pages:
        return
    
    for title, member in category_page.categorymembers.items():
        if title in visited_titles:
            continue
        visited_titles.add(title)
 
        if member.ns == wikipediaapi.Namespace.CATEGORY:
            recurse_category(member, max_pages=max_pages, depth=depth+1)
        elif member.ns == wikipediaapi.Namespace.MAIN:
            geo_pages.append(title)
            print(f"\r{len(geo_pages)}/{max_pages}\t{title}", end="")
            if len(geo_pages) > max_pages:
                break


def main():
    for cat in geo_categories:
        print(f"Processing category: {cat}")
        category_page = wiki.page("Category: " + cat)
        recurse_category(category_page, max_pages=1000000)


    print(f"Total unique geography-related articles collected: {len(geo_pages)}")

    with open("geography_page_titles.txt", "w", encoding="utf-8") as f:
        for title in geo_pages:
            f.write(title + "\n")
            
    for title in tqdm(geo_pages):
        content_path = os.path.join("colbert_qa_geography/data/database_wikipedia", title.replace("/", "").replace(" ", "_"))
        if os.path.exists(content_path):
            continue
        page = wiki.page(title)
        if page.exists():
            content = page.text
            try:
                with open(content_path, "w") as f_content:
                    f_content.write(content + "\n")
            except:
                print(f"Fail: {title}")


def get_contents(title_file):
    with open(title_file, "r") as f:
        titles = f.readlines()
    titles = [title.strip() for title in titles]
    for title in titles:
        content_path = os.path.join("colbert_qa_geography/data/database_wikipedia", title.replace("/", "").replace(" ", "_"))
        if os.path.exists(content_path):
            continue
        page = wiki.page(title)
        if page.exists():
            content = page.text
            with open(content_path, "w") as f_content:
                f_content.write(content + "\n")
        else:
            print(f"Page {title} does not exist")
        time.sleep(1)


if __name__ == "__main__":
    # main()
    get_contents("geography_page_titles.txt")