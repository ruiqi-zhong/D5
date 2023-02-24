import json

discoveries = json.load(open('sec5/discoveries.json'))

# the number 128th problem in openD5
idx = 128
problem = discoveries[128]

print('====== Research Goal ======= ')
print(problem['research goal'])

print('Significant discoveries that are more often true on Corpus A')
print(problem['+'].keys())

print('Significant discoveries that are more often true on Corpus B')
print(problem['-'].keys())

discovery = 'mentions children or family'
print('====== Discovery: %s =======' % discovery)
print('Here are the relevant statsitics, V\' and p-value')
print(problem['-'][discovery])

