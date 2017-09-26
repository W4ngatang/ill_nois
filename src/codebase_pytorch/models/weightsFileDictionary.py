

# juankyFolder = '/Users/juanperdomo/juanky/model_weights/'
juankyFolder = "/n/regal/ysinger_group/model_weights/"

juankyWeights = {
    'vgg11': juankyFolder + 'vgg11-bbd30ac9.pth',
    'vgg13': juankyFolder + 'vgg13-c768596a.pth',
    'vgg16': juankyFolder + 'vgg16-397923af.pth',
    'vgg19': juankyFolder + 'vgg19-dcbb9e9d.pth',
    'vgg11_bn': juankyFolder + 'vgg11_bn-6002323d.pth',
    'vgg13_bn': juankyFolder + 'vgg13_bn-abd245e5.pth',
    'vgg16_bn': juankyFolder + 'vgg16_bn-6c64b313.pth',
    'vgg19_bn': juankyFolder + 'vgg19_bn-c79401a0.pth',
    'resnet18': juankyFolder + 'resnet18-5c106cde.pth',
    'resnet34': juankyFolder + 'resnet34-333f7ec4.pth',
    'resnet50': juankyFolder + 'resnet50-19c8e357.pth',
    'resnet101': juankyFolder + 'resnet101-5d3b4d8f.pth',
    'resnet152': juankyFolder + 'resnet152-b121ed2d.pth',
    'densenet121': juankyFolder + 'densenet121-a639ec97.pth',
    'densenet169': juankyFolder + 'densenet169-b2777c0a.pth',
    'densenet201': juankyFolder + 'densenet201-c1103571.pth',
    'densenet161': juankyFolder + 'densenet161-8d451a50.pth',
    'squeezenet1_0': juankyFolder + 'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': juankyFolder + 'squeezenet1_1-f364aa15.pth',
    'alexnet': juankyFolder + 'alexnet-owt-4df8aa71.pth'
}

if __name__ == "__main__":
    import pickle

    with open('juankyWeightLocations.pickle', 'wb') as handle:
        pickle.dump(juankyWeights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print pickle.load(open('juankyWeightLocations.pickle', "rb"))