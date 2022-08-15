num = 10000
input_path = "D:/chat.txt"
output_path = "./Datas/chatbot.txt"

f_in = open(input_path, encoding='utf-8')
txt = []

count = 0
for line in f_in:
    if count < num:
        txt.append(line.split('\n'))
    count = count + 1

f_out = open(output_path, 'w', encoding='utf-8')

seg = 1
for line in txt:
    f_out.write(eval(str(line).replace("[", "").replace("]", "").replace(",", "")))
    f_out.write('\n')
    if seg != 0 and seg % 2 == 0:
        f_out.write('\n')
    seg = seg + 1

