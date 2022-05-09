from text_processor import TextProcessor 

def test_tokenize():
    ''' Tests the tokenize_links() function '''

    processor = TextProcessor()

    assert processor.tokenize("") == []
    assert processor.tokenize("[[]]") == []
    assert processor.tokenize("7") == ["7"]
    assert processor.tokenize("[a] !!") == ["a"]    
    
    assert processor.tokenize("[[Category:Computer Science]]") == ['[[category:computer science]]']
    assert processor.tokenize("Category:Computer Science") == ["category", "computer",  "science"]
    assert processor.tokenize("[[US Colleges|Washington and Lee]]") == ["[[us colleges|washington and lee]]"]
    assert processor.tokenize("US Colleges|Washington and Lee") == ["us", "colleges", "washington", "and", "lee"]

    text = "A computer science topic involves [[linear algebra]], which is hard." 
    assert processor.tokenize(text) == ["a", "computer", "science", "topic", "involves", "[[linear algebra]]", "which", "is", "hard"]


def test_is_link():
    ''' Tests the is_link() function '''

    processor = TextProcessor()

    assert processor.is_link("a") == False
    assert processor.is_link("[[linear algebra]]") == True


def test_is_stop_word():
    ''' Tests the is_stop_word() function '''

    processor = TextProcessor()

    assert processor.is_stop_word("a") == True
    assert processor.is_stop_word("the") == True
    assert processor.is_stop_word("it") == True
    assert processor.is_stop_word("from") == True

    assert processor.is_stop_word("computer") == False


def test_stem_word():
    ''' Tests the stem_word() function '''

    processor = TextProcessor()
    
    assert processor.stem_word("involve") == "involv"
    assert processor.stem_word("involves") == "involv"
    assert processor.stem_word("involving") == "involv"
    assert processor.stem_word("uninvolved") == "uninvolv"
    assert processor.stem_word("easiest") == "easiest"

# function calls!
test_tokenize()
test_is_link()
test_stem_word()
test_is_stop_word()