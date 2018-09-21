from bs4 import BeautifulSoup
import csv
import urllib.request

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

    main = soup.find("div", {"id": "main"})
    members = main.find_all("li", {"class": "compact"})
    i = 0
    members_dict = {}
    for member in members:
        link = member.find('a')
        name = link.get_text()
        name_list = name.split(' ')
        last = ''
        for n in name_list:
            try: 
                if n[-1] == ',':
                    last = n[:-1]
                    break
            except:
                continue
        number = link.get('href').split('/')[-1]
        image_link = "http://bioguide.congress.gov/bioguide/photo/" + number[0] + "/" + number + ".jpg"
        try:
            p = open('images/'+number+'.jpg','wb')
            p.write(urllib.request.urlopen(image_link).read())
            p.close()
            f.writerow([last, name, number, image_link, 'yes'])
        except:
            f.writerow([last, name, number, image_link, 'no'])
            i += 1
    print(i)

def run(csv_file, html_files):
    f = csv.writer(open(csv_file, "w"))
    f.writerow(["Last", "Name", "ID", "Image Link", "Image"])    # Write column headers as the first line

    for html_file in html_files:
        scrape_image_links(f, html_file)


if __name__ == '__main__':
    csv_file = "115_congress.csv"
    html_files = ['congress1.html', 'congress2.html', 'congress3.html']

    run(csv_file, html_files)