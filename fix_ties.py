import csv

with open('submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Sort by score desc, then candidate_id asc for tie-breaking
rows.sort(key=lambda x: (-float(x['score']), x['candidate_id']))

# Re-assign ranks
for i, row in enumerate(rows):
    row['rank'] = str(i + 1)

with open('submission.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate_id','rank','score','reasoning'])
    writer.writeheader()
    writer.writerows(rows)

print('Fixed! Now run: python validate_submission.py submission.csv')
