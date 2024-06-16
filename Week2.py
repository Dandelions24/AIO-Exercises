#Ex1
import math
num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1]
k = 2
result = []
sub_list = []

for element in num_list:
  sub_list.append(element)

  if len(sub_list) == k:
    result.append(max(sub_list))
    sub_list.pop(0)

print(result)


#Ex2
character_statistic = {}
word = 'dandelion'

for char in word:
  if char in character_statistic:
        character_statistic[char] += 1
  else:
        character_statistic[char] = 1

print(character_statistic)


#Ex3
with open('/content/Quỹ ETF.docx', encoding='latin-1') as file:
    document = file.read()
    print(type(document))

words = document.split()
counter = {}
for word in words:
    if word in counter:
        counter[word] += 1
    else:
        counter[word] = 1
print(counter)


#Ex4
def levenshtein_distance (token1, token2):
  distances = [[0]*(len(token2)+1) for i in range(len (token1)+1)]

  for t1 in range (len (token1) + 1):
    distances [t1][0] = t1

  for t2 in range(len(token2) + 1):
    distances [0] [t2] = t2

  for t1 in range(1, len (token1) + 1):
    for t2 in range(1, len(token2) + 1):
      if (token1[t1-1] == token2 [t2-1]):
        distances [t1] [t2] = distances [t1 - 1][t2 - 1]
      else:
        a = distances [t1][t2 - 1]
        b = distances [t1 - 1] [t2]
        c = distances [t1 - 1] [t2 - 1]

        if (a <= b and a <= c):
          distances [t1] [t2] = a + 1
        elif (b <= a and b <= c):
          distances [t1] [t2] = b + 1
        else:
          distances [t1] [t2] = c + 1

  return distances[len(token1)][len(token2)]

assert levenshtein_distance("hi", "hello") == 4
print (levenshtein_distance("hola", "hello"))