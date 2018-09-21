from bs4 import BeautifulSoup
import csv
import urllib.request
import unidecode

# image = "http://bioguide.congress.gov/bioguide/photo/" + bioguide[0] + "/" + bioguide + ".jpg"
# https://www.congress.gov/member/ralph-abraham/A000374
# https://programminghistorian.org/lessons/intro-to-beautiful-soup
# https://github.com/unitedstates/congress-legislators/issues/160
# https://github.com/unitedstates/congress-legislators/tree/master/scripts

# import requests
# r = requests.get('https://www.gpo.gov/fdsys/bulkdata/')
# soup = BeautifulSoup(r.content)


def scrape_image_links(f, html_file):
    soup = BeautifulSoup(open(html_file))

    tables = soup.find_all("tbody")
    
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            try:
                last = row.find("td", {"class": "name short"}).find('a').get_text()
                try:
                    last = last.split(' ')[1]
                    print(last)
                except:
                    pass

                last = unidecode.unidecode(last)
                name = row.find("td", {"class": "name long"}).find('a').get_text()
                party = row.find("td", {"class": "party"}).find('div').get_text()
                state = row.find("td", {"class": "state"}).get_text()
                trump = row.find("td", {"class": "score agree num"}).find('div').get_text()
                f.writerow([last, name, party, state, trump])
            except:
                break

def run(csv_file, html_files):
    f = csv.writer(open(csv_file, "w"))
    f.writerow(["Last Name", "Name", "Party", "State", "Trump Score"])    # Write column headers as the first line

    for html_file in html_files:
        scrape_image_links(f, html_file)


if __name__ == '__main__':
    csv_file = "trump_congress.csv"
    html_files = ['trump_congress.html']

    run(csv_file, html_files)