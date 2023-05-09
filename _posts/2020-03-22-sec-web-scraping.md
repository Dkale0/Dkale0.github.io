---
layout: post
title:  "SEC Filings Web Scraping"
author: darsh
categories: []
image: assets/images/sec_web_scraping/bs4_scraping_t.png
---



In this project we will make extracting information from the SEC Filings (10-K and 10-q) easily accessible and automate the process of retrieving only necessery data since these filings can be filled with information that may not be relavant. There are two classes we create to encapsulate the fuctionality for web scraping SEC filing from the EGAR Database and Parsing these filings that are in xml form. Note: there is a known [403 Forbidden error](https://github.com/jadchaar/sec-edgar-downloader/issues/77) in 2021 (unfixed, solution explained later) that prevents us from getting the filings in the xml form, which would have been ideal since the format is consistent across different filings for each company and make the extraction of tables very easy.

### Import Libraries


```python
import requests, json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys ## all keyboard keys imported
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import os, time
from bs4 import BeautifulSoup
import urllib
import shutil
import pandas as pd
import copy
import warnings
```

### Scrape Filings

The Scrape_filing class has the follwing methods:
##### __init__(self, driver_path=None, teardown=False)
* Sets driver_path to working directory. Note the user must first download a chrome driver from [this](https://chromedriver.chromium.org/downloads) site inorder to use selenium. This will also set download path to the same directory the chrome driver is in + "\\path_files", where files needed to access these filings will be stored. 
  
##### __exit____exit__(self, exc_type, exc_val, exc_tb)
* This method will quit the chrome tab used by selenium once the class is no longer in use. 
    
##### find_file(self, name, path)
* This method returns the file path after iterating through the giver directory. Will be used to find files in the path_files directory.
  
##### find_cik(self, company="BlackRock Inc.")
* This method opens the link "https://www.sec.gov/edgar/searchedgar/cik.htm" and inputs the arguments given for company and extracts the CIK (Central Index Key or CIK number is a number given to an individual, company, or foreign government by the United States SEC).
  
##### find_filing_address(self, cik, year=2020, quarter=1, only_10k=False)
* The parameters for this method are the CIK number (which we found with the method above), year, quarter, and boolean for if we want the 10k. The EDGAR databse stores the urls for all the companies that filed for a particular quarter in a particular year, on this page from the edgar search api: "https://www.sec.gov/Archives/edgar/full-index/{year})}/QTR{quarter)}". We download this file for future use into the folder we created for path files when we instantiate the class. Note that we do not use Selenium in headless mode here, which means this process of extracting the CIK will be slow and inefficient. The reason for using Selenium is for me to get familiarized with this library. From this we must extract the particular filing (10-k/q) that we are looking for given our CIK. There are many intricacies to getting to this step due to the way the EDGAR database is structured, and the comments in our code provide further explanation to this.

##### find_10k_address(self, cik, year):
* This method simply relies on the method above, however it calls it on the path_files for each quarter of a year since companies may not necessarily release their 10k's (annual filings) in the same quarter.

##### def filing_xml_form(self, file_add):
* Unfortunately when trying to access the filings in xml form, we get a [403 Forbidden error](https://github.com/jadchaar/sec-edgar-downloader/issues/77). The response from the SEC webmaster email for this issue was to use the form: response = requests.get(url, headers={'User-Agent': 'Company Name:idk@uu.uu) when requesting the json from their api. I did not implement this fix since I am unsure what the case is for somone not affiliated with a company. Instead we will try to parse the html text file for the tables, which will be much harder.




```python
class Scrape_Filing(webdriver.Chrome):
    def __init__(self, driver_path=None, teardown=False):
        if driver_path == None:
            driver_path = os.path.abspath(os.getcwd()) ## Gets current working directory
        self.driver_path = driver_path
        self.teardown = teardown
        os.environ['PATH'] += self.driver_path
        options = webdriver.ChromeOptions()
        #options.add_experimental_option('excludeSwitches', ['enable-logging']) ## stops the warnings related to reading file descriptors logs
        
        download_path = driver_path + '\\path_files'   # set path for when we download the filing to minimize requests sent to server when we parse it 
        self.download_path = download_path
        preferences = {
            "download.default_directory":download_path
            
        }
        options.add_experimental_option("prefs", preferences)
       
        super(Scrape_Filing, self).__init__(options=options) ## use super to instantiate the webdriver.chrome class
        self.implicitly_wait(15)
        # self.maximize_window()

    def __exit__(self, exc_type, exc_val, exc_tb): ## method to close the chrome window
        if self.teardown:
            self.quit()
            
    def find_file(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)
 
        
    def find_cik(self, company="BlackRock Inc."):
        self.get("https://www.sec.gov/edgar/searchedgar/cik.htm")
        company_element = self.find_element_by_id("company")
        company_element.send_keys(company)
        submit_element = self.find_element_by_class_name("search-button")
        submit_element.click() 
       
        try:

            table_element = self.find_element_by_css_selector("table[summary='Results of CIK Lookup']")
            rows = table_element.find_elements(By.TAG_NAME, "tr")

            td_row = rows[0].find_elements(By.TAG_NAME, "td") ## d_row has 2 rows
            
            pre_list = td_row[1].find_elements(By.TAG_NAME, "pre") ## there are 2 <pre> tags in 2nd rows
            cik = pre_list[1].find_elements(By.TAG_NAME, "a")[0].text ## we access the very first <pre> tag and the first a tags text is our cik 
            return cik
        except Exception as e:
            print("try a different name")
            
    def find_filing_address(self, cik, year=2020, quarter=1, only_10k=False): ## by default finds 10k, otherwise choose from quarter=1-4
        new_filename = f"master_{year}_{quarter}.txt"
        if self.find_file(new_filename, self.download_path): ## To prevent re downloading files, and make the program more efficient
            pass
            #print("Found File")
        else:
        
            master_idx_url = f"https://www.sec.gov/Archives/edgar/full-index/{str(year)}/QTR{str(quarter)}/"
            self.get(master_idx_url)
            table_element = self.find_element_by_css_selector("table[summary='heding']")
            rows_element = table_element.find_elements(By.TAG_NAME, "tr")
            master_idx_link = rows_element[11].find_element(By.TAG_NAME, "a")
            master_idx_link.click()
            ## sleep to let file download
            time.sleep(2)


            filename = max([self.download_path + "\\" + f for f in os.listdir(self.download_path)],key=os.path.getctime)
            print(filename)

            shutil.move(filename,os.path.join(self.download_path,new_filename))
            
        ## strip the preceding zeroes in the string cik
        stripped_cik = cik.lstrip("0") 
        add_10q = []
        add_10k = []
        with open(self.download_path + "\\" + new_filename) as fp:
            for line in fp:
                if stripped_cik in line:
                    if "10-Q" in line:
                        add_10q.append(line) 
                    if "10-K" in line:
                        add_10k.append(line)
                        
        if add_10k == [] and add_10q == []:
            print("No filings found")
            return None
                        
        return_address = []
        
        if only_10k == True:
            if add_10k == []:
                return None
            else:
                for line in add_10k:
                    return_address.append(line.split('|')[4])
                complete_address = "https://www.sec.gov/Archives/" + return_address[0].strip("\n")
                ## for now only returns complete address of first filing we find that is either a 10-K or 10-Q
                return complete_address 
            
                        
        ## Single quarter archive has multiple 10-Q, or both 10-K and 10-Q reports for some company(ie. some international companies)
        for line in add_10q:
            return_address.append(line.split('|')[4])
            
        complete_address = "https://www.sec.gov/Archives/" + return_address[0].strip("\n")
        return complete_address ## for now only returns complete address of first filing we find that is either a 10-K or 10-Q
            
        
      
    def find_10k_address(self, cik, year): ## Since companies may release their 10k's in different quarters
        for i in range(1, 5):
            address_10k = self.find_filing_address(cik=cik, year=year, quarter=i, only_10k=True)
            if address_10k != None:
                return address_10k
        return None
    
    def filing_xml_form(self, file_add):
        # 
        json_url = file_add.replace('-','').replace('.txt','/index.json')
        print(json_url)
        #son_decoded = requests.get(json_url).json()
        response = requests.get(json_url)
        response.raise_for_status()  # raises exception when not a 2xx response
        if response.status_code != 204:
            json_decoded = response.json() # We get a 403 error with a potential fix above
               
        for file in json_decoded['directory']['item']:
             if file['name'] == 'FilingSummary.xml':
                xml_ad = base_url + json_decoded['directory']['name'] + "/" + file['name']
                return xml_ad  
        
        
        
        
                        
```


```python
MicrosoftCorp10k = Scrape_Filing()
ms_cik = MicrosoftCorp10k.find_cik("Microsoft Corp") # use abbreviations such as Corp, Ltd, Inc, etc.
print(ms_cik)
file_ad = MicrosoftCorp10k.find_10k_address(ms_cik, 2019)

```

    0000789019
    


```python
print(file_ad)
ms_xml_ad = MicrosoftCorp10k.filing_xml_form(file_ad)
print(ms_xml_ad)
```

    https://www.sec.gov/Archives/edgar/data/789019/0001564590-19-027952.txt
    https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/index.json
    


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    <ipython-input-4-ffaeca76ee6b> in <module>
          1 print(file_ad)
    ----> 2 ms_xml_ad = MicrosoftCorp10k.filing_xml_form(file_ad)
          3 print(ms_xml_ad)
    

    <ipython-input-2-f5dde2ed0a7f> in filing_xml_form(self, file_add)
        122         #son_decoded = requests.get(json_url).json()
        123         response = requests.get(json_url)
    --> 124         response.raise_for_status()  # raises exception when not a 2xx response
        125         if response.status_code != 204:
        126             json_decoded = response.json()
    

    ~\anaconda3\envs\ml\lib\site-packages\requests\models.py in raise_for_status(self)
        951 
        952         if http_error_msg:
    --> 953             raise HTTPError(http_error_msg, response=self)
        954 
        955     def close(self):
    

    HTTPError: 403 Client Error: Forbidden for url: https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/index.json



```python
class ParseFiling():
    def __init__(self):
        self.filing = dict()
        self.filing['sec_header_content'] = {}
        self.filing['filing_documents'] = None
        self.test = 0
        
    def retrieve_filing(self, file_address):
        response = requests.get(file_address)
        filing = BeautifulSoup(response.content, 'lxml')
        sec_header_tag = filing.find('sec-header')
        

        
        display(sec_header_tag)
        
        # find condendsed consolidated statements of financial condition
        #<table border="0" cellspacing="0" cellpadding="0" style="margin:auto;border-collapse:collapse; width:100%;">
        print("???")
        i = 0
#         for filing_document in filing.find('document'):
#             document_filename = filing_document.filename.find(text=True, recursive=False).strip()
#             #print(i, ": ", document_filename)
#             display(document_filename) 
#             i+=1
#             #master_document_dict[document_id]['document_filename'] = document_filename
            
    def retrieve_xml(self, file_address):
        base_url = xml_summary.replace('FilingSummary.xml', '')
        content = requests.get(xml_summary).content
        soup = BeautifulSoup(content, 'lxml')
        # find the 'myreports' tag because this contains all the individual reports submitted.
        reports = soup.find('myreports')
        master_reports = []
        # loop through each report in the 'myreports' tag but avoid the last one as this will cause an error.
        for report in reports.find_all('report')[:-1]:
            #dictionary to store all the different parts we need.
            report_dict = {}
            report_dict['name_short'] = report.shortname.text
            report_dict['name_long'] = report.longname.text
            report_dict['position'] = report.position.text
            report_dict['category'] = report.menucategory.text
            report_dict['url'] = base_url + report.htmlfilename.text
           
def xml_table(self, )
                
    
```


```python
BlackRock10k = Scrape_Filing()
#cik = BlackRock10k.find_cik("BlackRock Inc.")
cik = '0001364742'
file_address = BlackRock10k.find_filing_address(cik, year=2019, quarter=2)
```


```python

```
