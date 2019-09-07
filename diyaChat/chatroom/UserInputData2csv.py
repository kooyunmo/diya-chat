from models import UserInputDataset

qas = UserInputDataset.objects.all()

for obj in qas:
    print("%s,%s,1" % (obj.question, obj.answer))
