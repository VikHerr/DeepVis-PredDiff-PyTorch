

def get_normalize_transform(dataset_name):
    norm_dict = {
        "01_mnist_fashion": transforms.Normalize((0.3201, 0.3182, 0.3629), (0.1804, 0.3569, 0.1131)),
        "02_cifar10": transforms.Normalize((0.4881, 0.4660, 0.3994), (0.2380, 0.2322, 0.2413)),
        "03_sen12ms": transforms.Normalize((0.1674, 0.1735, 0.2059), (0.1512, 0.1152, 0.1645)),
        "04_RSICD": transforms.Normalize((0.3897, 0.4027, 0.3715), (0.2050, 0.1920, 0.1934)),
        "05_xView2": transforms.Normalize((0.3292, 0.3408, 0.2582), (0.1682, 0.1408, 0.1296)),
        "06_So2SatLCZ42": transforms.Normalize((0.2380, 0.3153, 0.5004), (0.0798, 0.1843, 0.0666)),
    }
    return norm_dict[dataset_name]