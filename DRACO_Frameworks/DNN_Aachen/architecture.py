from keras import optimizers

architecture = {}

architecture["4j_ge3t"] = {
    "prenet_layer":             [100,100],
    "prenet_loss":              'categorical_crossentropy',
    "mainnet_layer":            [100,100],
    "mainnet_loss":             "kullback_leibler_divergence",
    "Dropout":                  0.30,
    "L2_Norm":                  1e-5,
    "batch_size":               5000,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "earlystopping_percentage": 0.01,
    "batchNorm":                False,
    }

architecture["5j_ge3t"] = {
    "prenet_layer":             [100,100],
    "prenet_loss":              'categorical_crossentropy',
    "mainnet_layer":            [100,100],
    "mainnet_loss":             "kullback_leibler_divergence",
    "Dropout":                  0.30,
    "L2_Norm":                  1e-5,
    "batch_size":               5000,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "earlystopping_percentage": 0.01,
    "batchNorm":                False,
    }

architecture["ge6j_ge3t"] = {
    "prenet_layer":             [100,100],
    "prenet_loss":              'categorical_crossentropy',
    "mainnet_layer":            [100,100],
    "mainnet_loss":             "kullback_leibler_divergence",
    "Dropout":                  0.30,
    "L2_Norm":                  1e-5,
    "batch_size":               5000,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "earlystopping_percentage": 0.01,
    "batchNorm":                False,
    }

def getArchitecture(cat):
    return architecture[cat]
