from pathlib import Path
from typing import Any, List
import requests
from newspaper import Article
from sentence_transformers import SentenceTransformer



from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the GOOGLE_KEY and GOOGLE_CX from environment variables
key = os.getenv("GOOGLE_KEY")
cx = os.getenv("GOOGLE_CX")


REQUEST_TIMEOUT = 5
NUM_SITES_TO_TRY = 6
MAX_PAGE_TEXT_LEN = 8000
MIN_PAGE_TEXT_LEN = 400
MIN_EXCERPTS = 4

# max length of top n excerpts to show to model
MAX_EXCERPTS_LEN = 1500

# keep adding lines until a section is at least this big.
MIN_SECTION_LEN = 300


class WebSearch:
    def __init__(self) -> None:
        pass

    def run(self, tool_input: str, debug=False):
        # search for information
        search_result_json = self._search(tool_input)

        excerpts: list[str] = []

        # visit top N results and extract their text content
        for i in range(0, NUM_SITES_TO_TRY):
            link = search_result_json['items'][i]['link']
            print(f'Reading {link}...')
            page_text = self._extract_text_from_url(link)

            if page_text is None:
                print(f'Skipping {link}')
                continue

            page_text = page_text.strip()

            if debug:
                print(f'found page of len: {len(page_text)}')

            page_text = page_text[:MAX_PAGE_TEXT_LEN]
            page_text = page_text.strip()

            if len(page_text) < MIN_PAGE_TEXT_LEN:
                print('Skipping (not enough text)')
                continue

            sections = split_text_into_sections(page_text, MIN_SECTION_LEN)
            excerpts.extend(sections)

        print(f'found {len(excerpts)} sections')

        # Remove newlines
        excerpts = [e.strip().replace('\n', ' ') for e in excerpts]

        return excerpts


    def get_tool_name(self) -> str:
        return 'WEB_SEARCH'

    def _extract_text_from_url(self, url: str) -> str | None:
        try:
            article = Article(url)
            article.download()
            article.parse()

            return article.text
        except:
            return None

    def _search(self, query) -> dict:

        query = query.replace('"', '')
        query = query + ' -site:youtube.com'

        url = 'https://customsearch.googleapis.com/customsearch/v1'
        params = {
            "key": key,
            "cx": cx,
            "q": query,
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            json = response.json()
            return json
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

def web_search(query: str, verbose=False):
    tool = WebSearch()
    return tool.run(query, verbose)

def find_closest_sections(sections: list[str], question: str) -> dict:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-mpnet-base-v2')
    corpus_embeddings: List[Tensor] | Any = model.encode(sections, convert_to_tensor=True)
    query_embedding: List[Tensor] | Any = model.encode([question], convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(corpus_embeddings, query_embedding)

    scored_sections = list(zip(sections, cosine_scores))
    scored_sections = sorted(scored_sections, key=lambda x: x[1], reverse=True)

    scores = [{'text': text, 'score': score.cpu().item()} for text, score in scored_sections]

    #Output the pairs with their score
    for i in range(4):
        print(f'[i] {scored_sections[i][1]}: {scored_sections[i][0]}')

    return scores

def split_text_into_sections(text: str, max_length: int = 400) -> list:
    # Split the text into sentences
    lines = text.splitlines()

    sentences = []

    for line in lines:
        parts = line.split('.')  # Split each line by period
        # Add the period back to the string
        parts_with_period = [bel + '.' for bel in parts if bel] 
        sentences.extend(parts_with_period)

    combined = []

    for sentence in sentences:
        if not combined:
            combined.append(sentence)
            continue

        prev = combined[-1]
        if len(prev) + len(sentence) < max_length:
            combined[-1] = f'{prev.strip()} {sentence.strip()}'
        else:
            combined.append(sentence)
    
    return combined