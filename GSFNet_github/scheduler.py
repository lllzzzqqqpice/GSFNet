

def schedule(epoch):
    if epoch<=200:
        lr=0.001
    else:
        lr=0.0001
    return lr
