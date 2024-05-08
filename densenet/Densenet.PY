import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn as nn

def main():
    #Преобразование данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Загрузка данных
    dataset = torchvision.datasets.ImageFolder(root='C:/Users/fedor/Desktop/diplom/dataset', transform=transform)
    shortDataset = torch.utils.data.Subset(dataset, range(10))

    #Разделенеие данных на тестовый и обучающий набор
    trainSize = int(0.8 * len(shortDataset))
    testSize = len(shortDataset) - trainSize
    trainDataset, testDataset = random_split(shortDataset, [trainSize, testSize])

    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=2)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=2)

    numClasses = 10

    model = models.densenet121(weights=None)

    #Замена полносвязанного слоя классификации на нужное количество классов
    numFtrs = model.classifier.in_features
    model.classifier = nn.Linear(numFtrs, numClasses)

    #Определение функции потери и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #Обучение
    numEpochs = 5
    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        for i, (inputs, labels) in enumerate(trainLoader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            if(i + 1) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{numEpochs}], Step [{i + 1}/{len(trainLoader)}], Loss:{runningLoss / (i + 1):.4f}]')
                runningLoss = 0.0
    print('Finished Training')        

    #Оценка точности
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, lables in testLoader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item() 
    accuracy = correct / total * 100
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')      

if __name__ == '__main__':
    main()