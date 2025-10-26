from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import time

url =  'https://www.tokopedia.com/msi-official-store/review'

driver = webdriver.Chrome()
driver.get(url)
time.sleep(5)  # Initial wait for the page to load

all_data = []

for i in range (10):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    containers = soup.find_all('article', attrs = {'class':'css-1pr2lii'})
    
    for container in containers:
        try:
            review = container.find('span', attrs = {'data-testid':"lblItemUlasan"}).text
            all_data.append(review)
        except AttributeError:
            continue
    
    time.sleep(4)  # Wait before clicking the next button
    driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Laman berikutnya"]').click()
    time.sleep(5)  # Wait for the next page to load

df = pd.DataFrame(all_data, columns=['review'])
df.to_csv('../data/reviews_raw.csv', index=False)
driver.quit()