import torch as th
from PIL import Image
from torchvision import transforms

def make_submission(net, has_3_channels: bool, data: str, transf, contains_labels=False):
    with th.no_grad():
        net.eval()
        acc = 0
        fout = open("data/submission.csv", "w")
        fout.write("id,label\n")
        fin = open("data/" + data + ".txt")
        lines = [i.split(',') for i in fin.readlines()]
        for line in lines:
            if contains_labels:
                name, label = line
                label = int(label)
            else:
                name = line[0][:-1]
            img = Image.open("data/" + data + "/" + name)
            img = transforms.ToTensor()(img).view(1, 1, 50, 50)

            if has_3_channels:
                img = th.stack([img, img, img]).view(1, 3, 50, 50)

            img = transf(img)

            result = net(img.cuda())

            predicted = result.argmax(dim=1)[0].item()

            if contains_labels:
                acc += (1 if predicted == label else 0)
            
            fout.write(name + "," + str(predicted) + "\n")

        fout.close()
        fin.close()

        if contains_labels:
            nr = len(lines)
            print(f"Accuracy: {round(acc / nr * 100, 2)}%")
