from tensorflow.keras import backend as kerasBackend
def L1(vectors):
    (featsA, featsB) = vectors
    sum = kerasBackend.abs(kerasBackend.sum(kerasBackend.abs(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(sum,kerasBackend.epsilon())

def L2(vectors):
    (featsA, featsB) = vectors
    sum = kerasBackend.sqrt(kerasBackend.sum(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(sum,kerasBackend.epsilon())

def MSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True)
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

def RMSE(vectors):
    (featsA, featsB) = vectors
    mean = kerasBackend.sqrt(kerasBackend.mean(kerasBackend.square(featsA - featsB), axis=1,keepdims=True))
    return kerasBackend.maximum(mean,kerasBackend.epsilon())

