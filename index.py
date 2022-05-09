import sys
import xml.etree.ElementTree as et
import math
import file_io
from text_processor import TextProcessor

class Index:
    ''' Class for the search Indexer '''

    def __init__(self):
        ''' Constructor for Index '''

        self.corpus = {} # dict mapping words -> number of documents containing this word
        self.all_max_counts = {} # dict mapping titles -> max number of occurences of any word 
        self.titles_to_ids = {} # dict mapping titles -> ids
        self.titles_to_processed_text = {} # dict mapping titles -> dicts mapping words -> counts
        self.all_relevances = {} # dict mapping words -> dicts mapping ids -> relevances
        self.page_weights = {} # dict mapping titles -> dicts mapping titles -> weights
        self.page_ranks = {} # dict mapping ids -> page ranks
        self.processor = TextProcessor() 


    def process_xml(self):
        ''' Processes every page in the XML and populates data structures for the index '''

        xml_filepath = sys.argv[1]
        root = et.parse(xml_filepath).getroot()
        all_pages = root.findall("page")

        for doc in all_pages:
            doc_id = int(doc.find("id").text)
            title = doc.find("title").text.strip().lower()
            self.titles_to_ids[title] = doc_id
            self.page_weights[title] = {}

            processed_text = self.process_text(title, doc.find("text").text)
            self.titles_to_processed_text[title] = processed_text

        self.calculate_relevance()
        self.calculate_page_ranks()

        self.titles_to_ids = { v:k for (k, v) in self.titles_to_ids.items() }


    def process_text(self, title: str, text: str) -> "dict[str, int]":
        '''
        Processes the text for a single document

        Parameters:
        title (str) -- title of the document being processed
        text (str) -- text to process

        Returns:
        (dict[str, int]) -- processed text
        '''

        all_tokens = self.processor.tokenize(text) + self.processor.tokenize(title)
        processed_text = {}
        max_count = 0

        for token in all_tokens:
            if self.processor.is_link(token):
                new_tokens = self.extract_tokens_from_link(token[2:-2], title)
                all_tokens.extend(new_tokens) 
                continue

            if not self.processor.is_stop_word(token):
                stemmed_word = self.processor.stem_word(token)
                self.all_relevances[stemmed_word] = {}
                
                if stemmed_word not in self.corpus:
                    self.corpus[stemmed_word] = 1 # SIDE EFFECT TO TEST!!!
                elif stemmed_word not in processed_text:
                    self.corpus[stemmed_word] = self.corpus[stemmed_word] + 1 # SIDE EFFECT TO TEST!!!

                processed_text[stemmed_word] = processed_text.get(stemmed_word, 0) + 1
                max_count = max(processed_text[stemmed_word], max_count)

        self.all_max_counts[title] = max_count

        return processed_text


    def extract_tokens_from_link(self, link: str, title: str) -> "list[str]":
        '''
        Produces a list of tokens from a given link and updates the graph of page linking relationships 

        Parameters:
        link (str) -- the link to be tokenized
        title (str) -- the title of the document containing the link

        Returns:
        (list[str]) -- list of tokens produced from the link
        '''

        if link.find("|") >= 0:
            left = link.split("|")[0] # links to this page title, non-tokenized
            right = self.processor.tokenize(link.split("|")[1]) # only want text right of the "|" as tokens
            self.page_weights[title][left] = None
        
            return right
        elif link.find("Category:") >= 0:
            tokens = ["category"] + self.processor.tokenize(link.split("Category:")[1]) 
            self.page_weights[title][link] = None

            return tokens
        else:
            tokens = self.processor.tokenize(link)
            self.page_weights[title][link] = None

            return tokens


    def calculate_relevance(self):
        ''' Calculates the relevance between all terms in the corpus and all documents '''
        
        doc_size = len(self.titles_to_ids)

        for word in self.corpus:
            idf = math.log(doc_size / self.corpus[word])

            for title in self.titles_to_ids:
                if word in self.titles_to_processed_text[title]:
                    doc_id = self.titles_to_ids[title]
                    tf = self.titles_to_processed_text[title][word] / self.all_max_counts[title]
                    self.all_relevances[word][doc_id] = tf * idf


    def calculate_page_ranks(self):
        ''' Calculates the PageRanks for all documents '''

        self.calculate_weights()
        n = len(self.titles_to_ids)
        delta = 0.001
        prev_row = { title: 0 for title in self.titles_to_ids }
        curr_row = { title: 1/n for title in self.titles_to_ids }

        while self.euclidean_distance(prev_row, curr_row) > delta:
            prev_row = curr_row.copy()

            for end_title in self.titles_to_ids:
                total = 0 

                for start_title in self.titles_to_ids:
                    total += self.page_weights[start_title][end_title] * prev_row[start_title] 

                curr_row[end_title] = total

        self.page_ranks = { self.titles_to_ids[k]:v for (k, v) in curr_row.items() }


    def calculate_weights(self):
        ''' Calculates and populates page_weights '''

        epsilon = 0.15
        n = len(self.titles_to_ids)

        for start_title in self.page_weights:
            nk = self.calculate_nk(start_title)
            links_to_nothing = False

            if nk == 0:
                links_to_nothing = True
                nk = n - 1
            
            for end_title in self.titles_to_ids:
                if start_title != end_title and \
                    (links_to_nothing or end_title in self.page_weights[start_title]):
                        self.page_weights[start_title][end_title] = (epsilon / n) + ((1 - epsilon) / nk)
                else:
                    self.page_weights[start_title][end_title] = (epsilon / n)
        

    def calculate_nk(self, title : str) -> int:
        ''' 
        Calculates nk for a given page, where nk the number of (unique) pages that k links to 
        
        Parameters:
        title (str) -- title of document to calculate for

        Returns:
        (int) -- value of nk for the given document
        '''

        count = 0
        edges = self.page_weights[title]

        for edge in edges:
            if edge in self.titles_to_ids and not edge == title:
                count += 1

        return count


    def euclidean_distance(self, v1: "dict[str, int]", v2: "dict[str, int]") -> float:
        ''' 
        Calculates the Euclidean distance between two vectors 

        Parameters:
        v1 (dict[str, int]) -- first vector
        v2 (dict[str, int]) -- second vector

        Returns:
        (float) -- Euclidean distance between the two vectors
        '''

        total = 0

        for key in v1:
            total += pow(v1[key] - v2[key], 2)

        return math.sqrt(total)


if __name__ == "__main__": 
    try:
        if len(sys.argv) != 5:
            print("Incorrect input, try again")
            quit()
        index = Index()
        index.process_xml()
        
        file_io.write_title_file(sys.argv[2], index.titles_to_ids)
        file_io.write_docs_file(sys.argv[3], index.page_ranks)
        file_io.write_words_file(sys.argv[4], index.all_relevances)
    except IOError:
        print("Incorrect input, try again")