import pytest
from query import Query

def test_calculate_scores():
    ''' Tests the calculate_scores() function '''

    query = Query()
    query.all_relevances["aa"] = { 1: 1.0986122886681098 }
    query.all_relevances["bb"] = { 2: 0.5493061443340549 }
    query.all_relevances["cc"] = { 2: 0.4054651081081644, 3: 0.27031007207210955 }
    query.all_relevances["dd"] = { 1: 0.0, 2: 0.0, 3: 0.0 }
    query.ids_to_titles = { 1: "AA", 2: "BB", 3: "CC" }
    query.page_ranks = { 1: 0.75, 2: 0.25, 3: 0 }

    # query.calculate_scores("AA", False) 
    # assert query.document_scores == { 1: 0, 2: 0, 3: 0 }

    query.calculate_scores(["aa"], False) 
    assert query.document_scores == { 1: 1.0986122886681098, 2: 0, 3: 0 } 
    
    query.calculate_scores(["aa"], True) 
    assert query.document_scores == { 1: 1.0986122886681098 * 0.75, 2: 0, 3: 0 } 

    query.calculate_scores(["aa", "dd"], False) 
    assert query.document_scores == { 1: 1.0986122886681098, 2: 0, 3: 0 } 

    query.calculate_scores(["aa", "aa"], False)
    assert query.document_scores == { 1: 1.0986122886681098 * 2, 2: 0, 3: 0 } 

    query.calculate_scores(["aa", "cc"], False)
    assert query.document_scores == {1: 1.0986122886681098, 2: 0.4054651081081644, 3: 0.27031007207210955 }