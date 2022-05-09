import pytest
from index import Index

def test_process_text():
    ''' Tests the process_text() function '''

    index = Index()
    index.corpus = { "aa": 1, "bb": 2 }
    index.page_weights["CC"] = {}
    index.all_max_counts = {}

    index.process_text("CC", "AA [[FF|AA]] CC")

    assert index.corpus == { "aa": 2, "bb": 2, "cc": 1 }
    assert index.all_max_counts == { "CC": 2 }


def test_extract_tokens_from_link():
    ''' Tests the extract_tokens_from_link() method '''
    
    index = Index()
    index.page_weights = {"A": {}}
    
    assert index.extract_tokens_from_link("US Colleges|Washington and Lee", "A") == ["washington", "and", "lee"]
    assert index.extract_tokens_from_link("Category:Computer Science", "A") == ["category", "computer", "science"]
    assert index.extract_tokens_from_link("Hammer", "A") == ["hammer"]

    assert index.page_weights["A"] == { "US Colleges": None, "Category:Computer Science": None, "Hammer": None }


def test_calculate_relevance():
    ''' Tests the calculate_relevance() function '''

    index = Index()
    index.corpus = { "aa": 1, "bb": 1, "cc": 2, "dd": 3 }
    index.titles_to_ids = { "AA": 1, "BB": 2, "CC": 3 }
    index.all_max_counts = { "AA": 1, "BB": 2, "CC": 3 }
    index.all_relevances = { "aa": {}, "bb": {}, "cc": {}, "dd": {} }

    index.titles_to_processed_text = \
    {
        "AA": {"aa": 1, "dd": 1 },
        "BB": {"bb": 1, "cc": 2, "dd": 2 }, 
        "CC": { "cc": 2, "dd": 3 }
    }

    # aa, bb, and cc has 0 tf for some documents but are not stored, dd has 0 idf
    index.calculate_relevance()
    assert index.all_relevances["aa"] == { 1: 1.0986122886681098 }
    assert index.all_relevances["bb"] == { 2: 0.5493061443340549 }
    assert index.all_relevances["cc"] == { 2: 0.4054651081081644, 3: 0.27031007207210955 }
    assert index.all_relevances["dd"] == { 1: 0.0, 2: 0.0, 3: 0.0 }

    
def test_calculate_page_ranks():
    delta = 0.000001
    index = Index()
    #testing PageRankExample2
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4}
    index.page_weights = {"A": {"C": None}, "B": {"D": None}, "C": {"D": None}, "D": {"A": None, "C": None}}

    index.calculate_page_ranks()
    assert index.page_ranks[1] - 0.20184346250214996 < delta
    assert index.page_ranks[2] - 0.03749999999999998 < delta
    assert index.page_ranks[3] - 0.37396603749279056 < delta
    assert index.page_ranks[4] - 0.3866905000050588 < delta

    #testing when every page only links to itself 
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    index.page_weights = {"A": {"A": None}, "B": {"B": None}, "C": {"C": None}, "D": {"D": None}, "E": {"E": None}}
    index.calculate_page_ranks()

    #if every page only links to itself, then the ranks should be same across all documents 
    for i in range(1, 6):
        assert index.page_ranks[i] == 0.2

    #testing when every page has no links
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    index.page_weights = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
    index.calculate_page_ranks()

    #if every page has no links, then all of the ranks should be the same across all documents
    #therefore, the page rank should be the same as testing when every page only links to itself
    for i in range(1, 6):
        assert index.page_ranks[i] == 0.2

    #testing when every page links to outside document outside corpus
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    index.page_weights = {"A": {"F": None}, "B": {"F": None}, "C": {"F": None}, "D": {"F": None}, "E": {"F": None}}
    index.calculate_page_ranks()

    #if every page links to outside the corpus, then the rank should be the same across all documents
    #therefore, the page rank should be same as the testing when there are no links 
    for i in range(1, 6):
        assert index.page_ranks[i] == 0.2
    #if there are multiple links to one page, then they are treated as a single link
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    index.page_weights = {"A": {"F": None, "F": None, "F": None}, "B": {"F": None}, \
         "C": {"F": None, "F": None, "F": None}, "D": {"F": None, "F": None, "F": None}, \
              "E": {"F": None, "F": None, "F": None}}
    for i in range(1, 6):
        assert index.page_ranks[i] == 0.2


def test_calculate_weights():
    ''' Test the calculate_weights() function '''

    index = Index()
    index.titles_to_ids = { "A": 1, "B": 2, "C": 3 }
    index.page_weights = { "A": {"B": None, "C": None }, "B": {}, "C": { "A": None } }

    index.calculate_page_ranks()
    
    assert index.page_weights["A"] == {'B': 0.475, 'C': 0.475, 'A': 0.049999999999999996}
    assert index.page_weights["B"] == {'A': 0.475, 'B': 0.049999999999999996, 'C': 0.475}
    assert index.page_weights["C"] == {'A': 0.9, 'B': 0.049999999999999996, 'C': 0.049999999999999996}

    assert pytest.approx(index.page_weights["A"]["A"]) == .05
    assert pytest.approx(index.page_weights["B"]["B"]) == .05
    assert pytest.approx(index.page_weights["C"]["C"]) == .05

    #testing PageRankExample2
    index.titles_to_ids = {"A": 1, "B": 2, "C": 3, "D": 4}
    index.page_weights = {"A": {"C": None}, "B": {"D": None}, "C": {"D": None}, "D": {"A": None, "C": None}}

    index.calculate_page_ranks()
    assert index.page_weights["A"] == {'C': 0.8875, 'A': 0.0375, 'B': 0.0375, 'D': 0.0375}
    assert index.page_weights["B"] == {'D': 0.8875, 'A': 0.0375, 'B': 0.0375, 'C': 0.0375}
    assert index.page_weights["C"] == {'D': 0.8875, 'A': 0.0375, 'B': 0.0375, 'C': 0.0375}
    assert index.page_weights["D"] == {'A': 0.46249999999999997, 'C': 0.46249999999999997, 'B': 0.0375, 'D': 0.0375}


def test_calculate_nk():
    ''' Tests the calculate_nk() function '''

    index = Index()
    index.titles_to_ids = { "A": 1, "B": 2, "C": 3, "D": 4 }
    index.page_weights = {"A": {}, "B": {}, "C": {}, "D": {}}

    assert index.calculate_nk("A") == 0

    #testing that pages that link to itself are ignored
    index.page_weights["A"]["A"] = None
    index.page_weights["B"]["B"] = None
    index.page_weights["C"]["C"] = None

    assert index.calculate_nk("A") == 0
    assert index.calculate_nk("B") == 0
    assert index.calculate_nk("C") == 0

    #testing that pages that link outside corpus are ignored
    index.page_weights["A"]["E"] = None
    index.page_weights["A"]["F"] = None
    index.page_weights["A"]["G"] = None
    assert index.calculate_nk("A") == 0

    index.page_weights["B"]["X"] = None
    index.page_weights["B"]["Y"] = None
    index.page_weights["B"]["Z"] = None
    assert index.calculate_nk("B") == 0

    index.page_weights["C"]["W"] = None
    index.page_weights["C"]["X"] = None
    index.page_weights["C"]["Y"] = None
    assert index.calculate_nk("C") == 0

    #testing that actual links are counted for
    index.page_weights["A"]["B"] = None
    index.page_weights["A"]["C"] = None
    assert index.calculate_nk("A") == 2

    index.page_weights["B"]["A"] = None
    index.page_weights["B"]["C"] = None
    assert index.calculate_nk("B") == 2

    index.page_weights["C"]["D"] = None
    assert index.calculate_nk("C") == 1


def test_euclidean_distance():
    ''' Tests the euclidean_distance() function '''

    index = Index()

    v1 = { "A": 1, "B": 2, "C": 3 }
    v2 = { "A": 1, "B": 2, "C": 3 }

    assert index.euclidean_distance(v1, v2) == 0

    v1 = { "A": 3, "B": -2, "C": 0 }
    v2 = { "A": 10, "B": 6, "C": 4 }

    assert index.euclidean_distance(v1, v2) - 11.357816 < 0.01


# function calls!
test_process_text()
test_extract_tokens_from_link()
test_calculate_relevance()
test_calculate_nk()
test_calculate_weights()