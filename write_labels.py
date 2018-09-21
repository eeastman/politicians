import csv

trump = open("id_trump.txt", "w")
age = open("id_age.txt", "w")
party = open("id_party.txt", "w")
region = open("id_region.txt", "w")

with open("Congress, White Male - Sheet1.csv") as f:
    reader = csv.reader(f)
    next(reader)
    total = 0
    for row in reader:
        row_id = row[5] + '.jpg '
        t = float(row[4][:-1])
        if t < 40:
            t = '0'
        elif t < 80:
            t = '1'
        else:
            t = '2'
        row_trump = row_id + t + '\n'
        trump.write(row_trump)
        rage = int(row[10])
        if rage <= 49:
            row_age = row_id + '0\n'
        elif rage <= 59:
            row_age = row_id + '1\n'
        elif rage <= 69:
            row_age = row_id + '2\n'
        else:
            row_age = row_id + '3\n'
        age.write(row_age)
        p = '1' if row[2] == 'D' else '0'
        party.write(row_id + p + '\n')
        region_row = row[11]
        if region_row == 'Northeast':
            f_region = '0\n'
        elif region_row == 'West':
            f_region = '1\n'
        elif region_row == 'South':
            f_region = '2\n'
        else:
            f_region = '3\n'
        region.write(row_id + f_region)
