import evaluate


rouge = evaluate.load('rouge')
rouge.add(predictions= "sentence 1", references= "sentence 2")
print(rouge.compute())

