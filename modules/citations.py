import re
from typing import List, Dict, Tuple

from modules.config import logger

def validate_citation_numbers(citation_numbers: List[int], max_docs: int) -> List[int]:
    """Validate that citation numbers are within the valid range"""
    return [num for num in citation_numbers if 1 <= num <= max_docs]

def process_citations(complete_answer: str, ranked_docs: List[dict]) -> Tuple[str, List[dict]]:
    """
    Extracts citation numbers from the answer, maps them to consecutive citation numbers,
    and returns the updated answer along with a list of citation sources.
    """
    citations = []
    seen_nums = set()
    citation_numbers = []
    for num_str in re.findall(r'\[(\d+)\]', complete_answer):
        num = int(num_str)
        if num not in seen_nums:
            seen_nums.add(num)
            citation_numbers.append(num)
    valid_citations = validate_citation_numbers(citation_numbers, len(ranked_docs))
    
    seen_urls = {}
    citation_map = {}
    current_num = 1
    for num in valid_citations:
        try:
            url = ranked_docs[num - 1]["page_source"]
            if url not in seen_urls:
                citation_map[num] = current_num
                seen_urls[url] = current_num
                citations.append({"url": url, "cite_num": str(current_num)})
                current_num += 1
            else:
                citation_map[num] = seen_urls[url]
        except IndexError:
            continue
    
    logger.debug(f"Citation numbers extracted: {citation_numbers}")
    logger.debug(f"Seen URLs mapping: {seen_urls}")

    def replace_citation(match):
        original = int(match.group(1))
        new_num = citation_map.get(original, original)
        url = next((c["url"] for c in citations if c["cite_num"] == str(new_num)), "")
        return f"[{new_num}]({url})" if url else f"[{new_num}]"

    updated_answer = re.sub(r'\[(\d+)\]', replace_citation, complete_answer)
    return updated_answer, sorted(citations, key=lambda x: int(x["cite_num"])) 