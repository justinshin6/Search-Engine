# Search README 
### Taiga Ji, Justin Shin 
## Overview 
- This project is a search engine that when given an input by the user, the program
will output the top-ten most relevant documents based off the input from the user. 
The search engine implementation is based off the PageRank algorithm for optimizing the search results based on relevance to the user query.  
## Known Bugs 
- At the beginning of our implementation, we were having trouble balancing time and space complexity. During our first few rounds of testing, we realized that we weren’t having the most optimal time complexity when indexing certain xml files, so we did a couple of changes (i.e., changing the corpus from a set to a dictionary, removing the calculating idf helper method, etc.) to optimize our time complexity to the point where medium sized xml files were indexing under 45 seconds. However, next space complexity was a problem as we realized that we were storing too many unnecessary values. One major switch that we made is that we didn’t store relevance values that equaled 0 as we realized that calculating the relevance values was taking up a lot of memory. Due to this one switch, we were able to decrease the amount of space needed tremendously to pass our tests. Therefore, as of now, there are no major known bugs in the implementation of our program after we optimized our time and space complexity. 
## Instructions For Use 
- The collections (called Wikis) of Wikipedia pages that are being searched over are called SmallWiki.xml, MedWiki.xml, and BigWiki.xml. To run the program you'll complete the following steps:
### 1. **In the terminal, input the following command:**
```
python3 index.py <XML filepath> <titles filepath> <docs filepath> <words filepath>
```
- This is called the indexing step of the search engine (further explained in the next section) where the documents inside the .xml file are prepared for querying by the user.
- Be certain that the Indexer needs to take in these inputs exactly **in this order** or else the search engine will not function. 
### 2. **After indexing, in the terminal, input the following command:**
```
python3 query.py <titles filepath> <docs filepath> <words filepath>
```
- This is called the querying step of the search engine (further explained in the next section) where the indexed documents are used to initiate the query. 
- It is very important in this section that the names of the filepaths match exactly what you inputted into the terminal in the indexing step. 
- Pagerank: If you want to incorporate the pagerank when trying to find the most relevant documents (further explained in next section), then input the following command:
```
python3 query.py --pagerank titles.txt docs.txt words.txt 
```
- Again, make sure that the names of the filepaths match exactly what you inputted into the terminal in the indexing step. 
### 3. **Input your query into the terminal**
- A search indicator will pop up in the terminal notifying the user to make a search query. 
### 4. **After inputting query, the top-ten most relevant documents will be outputted in order in the terminal.**
### 5. **Another search indicator will pop up for your next search.**
### 6. **Keep on using search engine until you input ":quit" into the search query which will terminate the engine.**  
## Description of Program 
## Indexing 
- The index.py file processes an xml document into a list of terms. Determines the relevance between the term and documents (pages), and determines the authority of each document. We will go through each of these steps one-by-one. 
### 1. **Processes an xml document into a list of terms:** 
- The indexer will process the xml file which is the name of the input file that the indexer will read and parse. The titles filepath will map document IDs to document titles. The docs filepath will store rankings computed by PageRank. The words filepath will store the relevance of documents to words 
- However, each word in the xml document has content that isn't relevant, so before querying, the indexer will remove irrelevant words such as stop words (i.e., ignoring words such as "a" and "the"), will tokenize the text (i.e., split the text into words and numbers, remove punctuation, etc.), and stem the words (reduce words to their root stems). 
### 2. **Determine relevance between the term and documents:**
- To score the relevance of a document to a query, we compare the two sequences of terms. Similarity metrics used by most practical search engines capture two key ideas: term frequency and inverse document frequency. 
- Term Frequency: A term that appears many times in a document is more likely to be more relevant than a term that appears fewer times. To normalize the frequency of counts across the docuemnts, we calcualte the term frequency by dividing the count of a term by the count of the most frequently used term in the same document. 
- Inverse Document Frequency: A term that occurs in fewer documents is likely to be a better signal of relevance when it occurs than a term that appears across many documents. We calculate the inverse document frequency for a term to be the natural log of the total number of documents divided by the number of documents that contain that term. 
- To score the final relevance of a document, we calculate the relevance score of a certain document to a certain term to be the normalized term frequency of that term in that document multiplied by the inverse document frequency of that term. 
### 3. **Determine the authority of each document:**
- PageRank: The PageRank algorithm, designed by the founders of Google, ranks pages based on the links among them (without considering the content). Pages with high scores are thought to be authoritative. Given a query, the most authoritative documents related to the words in the query are returned. These are the general principles when it comes to the PageRank Algorithm. 
- The more pages that link to a page, the more authoritative that page becomes. 
- The more authoritativethose pages are, the still more authoritative that certain page should be. 
- The fewer links those pages have to pages other than that certain page, the more authoritative that certain page should be. 
- The closer that certain page is to another page, the more the other page should influence the original page. 
- Each page's authority is a number between 0 and 1, where the total authority across all documents always equals 1. 
- After calculating the authorities and the ranks, the file.io file will write in the relevance and page ranks into the titles, docs, and words filepaths for use by the Querier.
## Querying 
- The second big piece of the search engine is the Querier. the Querier parses in arguments for the index files and an optional argument that says to use PageRank. It runs a REPL (Read-Eval-Print-Loop) that takes in and processes search queries. Finally, it scores documents against queries based on term relevance and PageRank (if specified) index files.
- After inputting the command in the terminal stated in the section above, the file.io file will read the information found in the indexed files for the query. Then the REPL will prompt and read in the user query in the terminal. The query will be answered by scoring its terms against every document and returning the titles of the documents with the top 10 scores. The previous steps will be repeated until the user types "quit:"
- When accounting for PageRank, we multiplied the pagerank by the term-relevance scores to output the most authoritative documents to the user. 
## Description of Features Failed to Implement
- In this project, we didn't fail any major features for our search engine. 
## Description of Testing and All System Tests
## Example of System Tests
- The first few batches of system of tests will include example querys from the TA example queries which will test general cases. 
### 1. **Baseball MedWiki Query**
- This query is the baseball query with the following index and query commands entered into the terminal: 
```
python3 index.py xml/MedWiki.xml titles.txt docs.txt words.txt 
python3 query.py titles.txt docs.txt words.txt 
```
- As shown, pagerank was not taken into account in this query. When the query "baseball" is entered into the search engine, the following top 10 documents show up:
```
1 oakland athletics
2 minor league baseball
3 miami marlins
4 fantasy sport
5 kenesaw mountain landis
6 out
7 october 30
8 january 7
9 hub
10 february 2
```
- This query exactly matches the TA example query where the most relevant document titles when it comes to the "baseball" query show up. 
- The next query is the baseball query including pagerank which has the same index command entered into the terminal above, but it has the following query command:
```
python3 query.py --pagerank titles.txt docs.txt words.txt
```
- Now, when the query "baseball" is entered into the search engine, the following 10 documents show up:
```
1 minor league baseball
2 kenesaw mountain landis
3 ohio
4 oakland athletics
5 miami marlins
6 february 2
7 netherlands
8 fantasy sport
9 out
10 kansas
```
- All ten of the documents in the following search results are found in the top 20 results found in the TA example query. 
### 2. **Fire MedWiki Query** 
- This query has the following index and query commands entered into the terminal:
```
python3 index.py xml/MedWiki.xml titles.txt docs.txt words.txt 
python3 query.py titles.txt docs.txt words.txt 
```
- As shown, pagerank was not taken into account in this query. When the query "fire" is entered into the search engine, the following top 10 documents show up:
```
1 firewall (construction)
2 pale fire
3 ride the lightning
4 g?tterd?mmerung
5 fsb
6 keiretsu
7 hephaestus
8 kab-500kr
9 izabella scorupco
10 justin martyr
```
- This query exactly matches the TA example query where the most relevant document titles when it comes to the "fire" query show up. 
- The next query is the fire query including pagerank which has the same index command entered into the terminal above, but it has the following query command:
```
python3 query.py --pagerank titles.txt docs.txt words.txt
```
- Now, when the query "fire" is entered into the search engine, the following 10 documents show up:
```
1 firewall (construction)
2 empress suiko
3 justin martyr
4 hephaestus
5 pale fire
6 new amsterdam
7 falklands war
8 hermann g?ring
9 guam
10 parmenides
```
- Nine of the documents in the following search results are found in the top 20 results found in the TA example query. 
### 3. **Cats MedWiki Query** 
- This query has the following index and query commands entered into the terminal:
```
python3 index.py xml/MedWiki.xml titles.txt docs.txt words.txt 
python3 query.py titles.txt docs.txt words.txt 
```
- As shown, pagerank was not taken into account in this query. When the query "cats" is entered into the search engine, the following top 10 documents show up:
```
1 kattegat
2 kiritimati
3 morphology (linguistics)
4 northern mariana islands
5 lynx
6 freyja
7 politics of lithuania
8 isle of man
9 nirvana (uk band)
10 autosomal dominant polycystic kidney
```
- This query exactly matches the TA example query where the most relevant document titles when it comes to the "cats" query show up. 
- The next query is the cats query including pagerank which has the same index command entered into the terminal above, but it has the following query command:
```
python3 query.py --pagerank titles.txt docs.txt words.txt
```
- Now, when the query "cats" is entered into the search engine, the following 10 documents show up:
```
1 netherlands
2 pakistan
3 morphology (linguistics)
4 northern mariana islands
5 kattegat
6 hong kong
7 normandy
8 isle of man
9 kiritimati
10 grammatical gender
```
- All of the 10 documents in the following search results are found in the top 20 results found in the TA example query. 

- The next set of tests will test base and edge cases that need to be accounted for. 
### 4. Testing Pages that Only link to Themselves 
- One of the special cases when accounting for the weights are links that link to themselves. In this case, a link from a page to itself is ignored. This query had the following commands into the terminal: 
```
python3 index.py xml/test_link_to_itself.xml titles.txt docs.txt words.txt
python3 query.py --pagerank titles.txt docs.txt words.txt
``` 
- The xml file that we indexed over is the test_link_to_itself.xml which had the following contents: 
```
<xml>
    <page><title>Title A</title><id>1</id><text>[[Title A]] Computer Science rocks. Computer Science is absolutely amazing.</text></page>
    <page><title>Title B</title><id>2</id><text>[[Title B]] This is a filler sentence.</text></page>
    <page><title>Title C</title><id>3</id><text>[[Title C]] Another sentence.</text></page>
    <page><title>Title D</title><id>4</id><text>[[Title D]] Cool.</text></page>
    <page><title>Title E</title><id>5</id><text>[[Title E]] Very nice!</text></page>
    <page><title>Title F</title><id>6</id><text>[[Title F]] Very very cool. </text></page>
    <page><title>Title G</title><id>7</id><text>[[Title G]] Another one. </text></page>
    <page><title>Title H</title><id>8</id><text>[[Title H]] DJ </text></page>
    <page><title>Title I</title><id>9</id><text>[[Title I]] Wowza. Computer science.</text></page>
    <page><title>Title J</title><id>10</id><text>[[Title J]] Another really long sentence.</text></page>

</xml>

```
- As you can see, the only links that are apparent are links to itself, so since these links are ignored, we only calculate the most relevant documents depending on the relevance scores. Therefore, if we search "computer science" for example, we get the following results:
```
search> computer science
1 title a
2 title i
```
- This makes sense as the A document and the I document were the only documents that contained the phrase "computer science," so the most relevant documents were the documents with the phrase in order of their id. 
- Another query that we tested with an xml file with documents that just link to themselves was "sentence" which had the following output: 
```
search> sentence
1 title b
2 title c
3 title j
```
- Once again, this makes sense also since documents B, C, and J were the only documents that had "sentence" in the text, so the most relevant documents were just the documents listed in order of id.
- All in all, this is exactly what we were looking for since we wanted links from a page to itself to be ignored, so the links had no effect on the most relevant documents. 
### 5. Testing xml file with no links
- To test our previous edge case (a page that links to itself is ignored), we can try inputting the same queries as before using the same xml file just without the links. Therefore, for this system test, we used the test_no_links.xml file which is exactly the same as the test_link_to_itself file except all of the links are removed: 
```
<xml>
    <page><title>Title A</title><id>1</id><text>Computer Science rocks. Computer Science is absolutely amazing.</text></page>
    <page><title>Title B</title><id>2</id><text>This is a filler sentence.</text></page>
    <page><title>Title C</title><id>3</id><text>Another sentence.</text></page>
    <page><title>Title D</title><id>4</id><text>Cool.</text></page>
    <page><title>Title E</title><id>5</id><text>Very nice!</text></page>
    <page><title>Title F</title><id>6</id><text>Very very cool. </text></page>
    <page><title>Title G</title><id>7</id><text>Another one. </text></page>
    <page><title>Title H</title><id>8</id><text>[DJ </text></page>
    <page><title>Title I</title><id>9</id><text>Wowza. Computer science.</text></page>
    <page><title>Title J</title><id>10</id><text>Another really long sentence.</text></page>

</xml>

```
- Therefore, since this is technically the same xml file as we used in test number #4, we should get the same query results for "computer science" and "sentence" as we did before, and that is exactly what happens. The following is an output of the following searches:
```
search> computer science
1 title a
2 title i
search> sentence
1 title b
2 title c
3 title j
```
- We can see that the search results exactly match the previous test which proves that we successfully ignored pages that link to themselves, and for tests #4 and #5, we are calculating the most relevant documents simply based off of relevance of the query. 
### 6. Testing xml file with links outside corpus 
- Another edge case that we need to account for is if there are links from an xml file to documents outside of the corpus. Like the edge case tests above, we need to ignore these extraneous links when calculating the PageRank. Therefore, to test this, we will use the test_link_outside_corpus.xml file which is the same xml file used in the above tests, but the links are all directed outside of the corpus:
```
<xml>
    <page><title>Title A</title><id>1</id><text>[[Title Z]] Computer Science rocks. Computer Science is absolutely amazing.</text></page>
    <page><title>Title B</title><id>2</id><text>[[Title Y]] This is a filler sentence.</text></page>
    <page><title>Title C</title><id>3</id><text>[[Title W]] Another sentence.</text></page>
    <page><title>Title D</title><id>4</id><text>[[Title V]] Cool.</text></page>
    <page><title>Title E</title><id>5</id><text>[[Title S]] Very nice!</text></page>
    <page><title>Title F</title><id>6</id><text>[[Title T]] Very very cool. </text></page>
    <page><title>Title G</title><id>7</id><text>[[Title Q]] Another one. </text></page>
    <page><title>Title H</title><id>8</id><text>[[Title P]] DJ </text></page>
    <page><title>Title I</title><id>9</id><text>[[Title M]] Wowza. Computer science.</text></page>
    <page><title>Title J</title><id>10</id><text>[[Title K]] Another really long sentence.</text></page>

</xml>
```
- Therefore, since this is technically the same xml file as we used in test number #5, we should get the same query results for "computer science" and "sentence" as we did before, and that is exactly what happens. The following is an output of the following searches:
```
search> computer science
1 title a
2 title i
search> sentence
1 title b
2 title c
3 title j
```
- We can see that the search results exactly match the previous test which proves that we successfully ignored pages that link to documents outside of the corpus.
### 7. Testing documents with multiple links in one page
- Another special case that we need to account for is when there are pages that hvae multiple links to the same page. In this case, we will consider those multiple links to be one singular link; therefore, to test this, we will use the test_multiple_links xml file which has pages where each page contains multiple links to a page:
```
<xml>
    <page><title>Title A</title><id>1</id><text>[[Title A]] [[Title A]] [[Title A]] Computer Science rocks. Computer Science is absolutely amazing.</text></page>
    <page><title>Title B</title><id>2</id><text>[[Title B]] [[Title B]] [[Title B]] This is a filler sentence.</text></page>
    <page><title>Title C</title><id>3</id><text>[[Title C]] [[Title C]] [[Title C]] Another sentence.</text></page>
    <page><title>Title D</title><id>4</id><text>[[Title D]] [[Title D]] Cool.</text></page>
    <page><title>Title E</title><id>5</id><text>[[Title E]] [[Title E]] Very nice!</text></page>
    <page><title>Title F</title><id>6</id><text>[[Title F]] [[Title F]] [[Title F]] [[Title F]] Very very cool. </text></page>
    <page><title>Title G</title><id>7</id><text>[[Title G]] [[Title G]] [[Title G]] Another one. </text></page>
    <page><title>Title H</title><id>8</id><text>[[Title H]] [[Title H]] DJ </text></page>
    <page><title>Title I</title><id>9</id><text>[[Title I]] [[Title I]] Wowza. Computer science.</text></page>
    <page><title>Title J</title><id>10</id><text>[[Title J]] [[Title J]] Another really long sentence.</text></page>

</xml>

```
- Therefore, since this is technically the same xml file as we used in test number #4, we should get the same query results for "computer science" and "sentence" as we did before, and that is exactly what happens. The following is an output of the following searches:
```
search> computer science
1 title a
2 title i
search> sentence
1 title j
2 title b
3 title c
```
- We can see that the search results exactly match the previous test which proves that we successfully accounted for the special case when there are multiple links to the same page. 
### 8. Testing No Results Query
- There is also the possibility that the query that we input into the terminal doesn't have any search results if we search for something that is outside of the corpus. When this occurs, an informative message needs to be printed out to indicate to the user that their search had no results. If we use the test_no_links.xml to test this, we have the following outputs:
```
search> Brown University
NO SEARCH RESULTS MATCHED YOUR QUERY. TRY AGAIN.
search> Classical Conditioning
NO SEARCH RESULTS MATCHED YOUR QUERY. TRY AGAIN.
search> Linear Algebra
NO SEARCH RESULTS MATCHED YOUR QUERY. TRY AGAIN.
```
- For the three respective search queries, we see that there are no relevant documents that match the queries, so the message "NO SEARCH RESULTS MATCHED YOUR QUERY. TRY AGAIN" is returned to notify the user. 
### 9. Testing Invalid Input in Query
- Another possibility is that the input into the terminal when indexing or querying doesn't match the criteria needed for the search engine to function. In this case, we outputted an informative message to tell the user that their input didn't match the requirements for the search engine. For instance, if this wrong input is inputted into the terminal:
```
python3 index.py xml/PageRankExample2.xml invalidArgument.txt
```
Then, the following will be outputted:
```
Incorrect input, try again
```
Furthermore, if the indexing went well, but the wrong input for the query is inputted into the terminal:
```
python3 query.py anotherInvalidArgument.txt
```
Then, the following will be outputted:
```
Incorrect input, try again
```

