# Book Scraping and Data Analysis using Scrapy



## Student Details



Name: Diya Tilwani

Student ID: 202618011



## Assignment Objective



Build a complete data pipeline by scraping book information to Scrape using Python and Scrapy, pre-process the collected data, generate visualizations, and derive meaningful insights.



## &#x20;Website Used



https://books.toscrape.com



## &#x20;Dataset



The scraper collects the data from the first five catalogue pages .



Extracted attributes:



1\. Title

2\. Category

3\. Price

4\. Rating

5\. Availability

6\. Product Description

7\. UPC

8\. Number of Reviews

9\. Product URL





## Project Workflow

## 

## Task 1 – Data Scraping



\* Developed a Scrapy spider.

\* Crawled the first five catalogue pages.

\* Visited every book page.

\* Extracted all required attributes.

\* Exported data to books\_raw.csv.

## 

## Task 2 – Data Preprocessing



1\. Removed duplicate books using UPC.

2\. Cleaned missing descriptions.

3\. Converted prices to numeric values.

4\. Converted ratings from text to integers.

5\. Extracted stock count.

6\. Created engineered features:

* description\_word\_count
* price\_band
* affordability\_score



The cleaned dataset was saved as books\_clean.csv.



## Task 3 – Visualization



Generated the following visualizations:



1\. Price Distribution

2\. Rating Distribution

3\. Average Price by Category

4\. Price vs Rating

5\.  Word Cloud using Book Descriptions

## 

## Task 4 – Analysis



Performed exploratory data analysis to identify:

1\. Category-wise pricing

2\. Rating patterns

3\. Stock availability

4\. Best-value books

5\. Dataset limitations



&#x20;Repository Contents

1.Scrapy Spider

2\. Raw Dataset

3\. Cleaned Dataset

4\. Preprocessing Script

5\. Visualization Script

6\. Generated Plots

7\. Summary Results



Libraries Used

1\. Python

2\. Scrapy

3\. Pandas

4\. NumPy

5\. Matplotlib

6\. WordCloud





## How to Run



Install dependencies

pip install -r requirements.txt





Run the spider

scrapy crawl books -o books\_raw.csv





Run preprocessing

python preprocessing.py



Run visualization

python visualization.py





