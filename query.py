from typing import IO
import file_io
import sys
from text_processor import TextProcessor

class Query:
    ''' Class for the search Querier '''

    def __init__(self):
        ''' Constructor for Query '''

        self.ids_to_titles = {} # dict mapping ids -> titles
        self.all_relevances = {} # dict mapping words -> dicts mapping ids -> relevances
        self.page_ranks = {} # dict mapping ids -> page ranks
        self.document_scores = {} # dict mapping ids -> scores


    def calculate_scores(self, processed_tokens: "list[str]", use_page_rank: bool):
        '''
        Calculates scores by summing the term-document scores for all terms in the query

        Parameters:
        processed_tokens (list[str]) -- all terms in the query
        use_page_rank (bool) -- whether to include pagerank or not in scoring
        '''

        for doc_id in self.page_ranks:
            score = 0

            for word in processed_tokens:
                if word in self.all_relevances:
                    score += self.all_relevances[word].get(doc_id, 0)

            if use_page_rank:
                self.document_scores[doc_id] = score * self.page_ranks[doc_id]
            else:
                self.document_scores[doc_id] = score


    def rank_documents(self) -> "list[int]":
        ''' 
        Prints and returns the 10 highest-scored documents matching with the query 
        
        Returns:
        (list[int]) -- list of highest-scored documents
        '''

        ranked_documents = []

        for i in range(0, min(len(self.ids_to_titles), 10)):
            max = 0 
            max_key = None

            for key in self.document_scores:
                if self.document_scores[key] > max:
                    max = self.document_scores[key]
                    max_key = key
            
            if max_key is None:
                if i == 0:
                    print("NO SEARCH RESULTS MATCHED YOUR QUERY. TRY AGAIN.")
                break
            else:
                print(i + 1, self.ids_to_titles[max_key])
                ranked_documents.append(self.ids_to_titles[max_key])
                self.document_scores.pop(max_key)

        return ranked_documents


###############################################################
########################### REPL ##############################
###############################################################

if __name__ == "__main__":
    try:
        q = Query()

        if len(sys.argv) == 5:
            file_io.read_title_file(sys.argv[2], q.ids_to_titles)
            file_io.read_docs_file(sys.argv[3], q.page_ranks)
            file_io.read_words_file(sys.argv[4], q.all_relevances)
        elif len(sys.argv) == 4:        
            file_io.read_title_file(sys.argv[1], q.ids_to_titles)
            file_io.read_docs_file(sys.argv[2], q.page_ranks)
            file_io.read_words_file(sys.argv[3], q.all_relevances)
        else:
            print("Incorrect input, try again")
            quit()
        
        query = input("search> ")
        processor = TextProcessor()

        while query != ":quit":
            all_tokens = processor.tokenize(query) 
            processed_tokens = [processor.stem_word(token) for token in all_tokens \
                if not processor.is_stop_word(token)]
            use_page_rank = False

            if "--pagerank" in sys.argv:
                use_page_rank = True

            q.calculate_scores(processed_tokens, use_page_rank)
            q.rank_documents()

            query = input("search> ")
    except IOError:
        print("Incorrect input, try again")