import pandas as pd
import sqlite3

# Load Excel file
df = pd.read_excel('Knowledge.xlsx', sheet_name='Knowledge')

# Connect to SQLite database
conn = sqlite3.connect('careers.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS careers (
    soc_code TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS career_subjects (
    soc_code TEXT,
    subject_id INTEGER,
    req_knowledge REAL,
    req_importance REAL,
    FOREIGN KEY (soc_code) REFERENCES careers(soc_code),
    FOREIGN KEY (subject_id) REFERENCES subjects(id),
    PRIMARY KEY (soc_code, subject_id)
)
''')

# Insert unique subjects
subjects = sorted(df['Element Name'].unique())
for i, sub in enumerate(subjects, 1):
    cursor.execute('INSERT OR IGNORE INTO subjects (id, name) VALUES (?, ?)', (i, sub))

# Insert unique careers
careers = df[['O*NET-SOC Code', 'Title']].drop_duplicates()
for _, row in careers.iterrows():
    cursor.execute('INSERT OR IGNORE INTO careers (soc_code, title) VALUES (?, ?)', 
                  (row['O*NET-SOC Code'], row['Title']))

# Insert career-subject requirements
subject_id_map = {name: i for i, name in enumerate(subjects, 1)}

# First, collect all the data in a dictionary
career_subjects_data = {}
for _, row in df.iterrows():
    soc_code = row['O*NET-SOC Code']
    subject = row['Element Name']
    subject_id = subject_id_map[subject]
    scale = row['Scale ID']
    value = row['Data Value']
    
    key = (soc_code, subject_id)
    if key not in career_subjects_data:
        career_subjects_data[key] = {'req_knowledge': None, 'req_importance': None}
    
    if scale == 'LV':
        career_subjects_data[key]['req_knowledge'] = value
    elif scale == 'IM':
        career_subjects_data[key]['req_importance'] = value

# Now insert all the data in a single pass
for (soc_code, subject_id), values in career_subjects_data.items():
    cursor.execute('''
        INSERT OR REPLACE INTO career_subjects 
        (soc_code, subject_id, req_knowledge, req_importance)
        VALUES (?, ?, ?, ?)
    ''', (soc_code, subject_id, values['req_knowledge'], values['req_importance']))

conn.commit()
conn.close()
print("Database created successfully.")