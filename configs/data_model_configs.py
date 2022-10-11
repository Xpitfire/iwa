from box import Box


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("7", "13"), ("12", "16"), ("9", "18"), ("6", "23")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("7", "18"), ("20", "30"), ("35", "31"), ("18", "23"), ("6", "19")]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("2", "7"), ("0", "6"), ("1", "6"), ("3", "8"), ("4", "5")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class MINI_DOMAIN_NET(object):
    def __init__(self):
        super(MINI_DOMAIN_NET, self).__init__()        
        self.class_names = ['squirrel', 'spreadsheet', 'dog', 'golf_club', 'tree']
        self.sequence_len = 128
        self.scenarios = [("real", "painting"), ("real", "quickdraw"), ("real", "sketch"), ("real", "clipart"), ("real", "infograph")]
        self.num_classes = 5
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        
        self.model = Box({
            "n_hidden": 512
        })
        
        self.backbone = Box({
            "model": "resnet18",
            "type": "Pretrained2D",
            "pretrained": True,
            "pretraining_dataset": "imagenet",
            "feature_layer": "layer4",
            "apply_avgpool": True,
            "avgpool": [1, 1],
            "trainable": False
        })
        
        self.dataloader = Box({
            "pretrained": True,
            "pretraining_dataset": "imagenet",
            "dataset": "DomainNet",
            "full_data_in_memory": True,
            "reset_and_reload_memory": True,
            "combined_source": False,
            "DomainNet": {
                "num_classes": 5,
                "selected_classes": [281, 278,  91, 131, 322],
                "num_workers": 8,
                "data_root": "data/MiniDomainNet",
                "image_size": 256,
                "crop": 244,
                "color_jitter_factor": 0.25,
                "rotation_degrees": 2,
                "scale": [0.7, 1.0],
                "test_dir": "test",
                "train_dir": "train",
                "data_dir": "data",
                "domains": ["real", "painting", "quickdraw", "sketch", "clipart", "infograph"],
                "clipart_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                "clipart_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
                "clipart_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt",
                "infograph_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                "infograph_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
                "infograph_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt",
                "painting_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                "painting_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
                "painting_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt",
                "quickdraw_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                "quickdraw_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
                "quickdraw_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt",
                "real_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                "real_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
                "real_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
                "sketch_data_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
                "sketch_train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
                "sketch_test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt"
            }
        })


class AMAZON_REVIEWS(object):
    def __init__(self):
        super(AMAZON_REVIEWS, self).__init__()        
        self.class_names = ['positive', 'negative']
        self.sequence_len = 128
        self.scenarios = [("books", "dvd"), ("books", "electronics"), ("books", "kitchen"),
                          ("dvd", "books"), ("dvd", "electronics"), ("dvd", "kitchen"),
                          ("electronics", "books"), ("electronics", "dvd"), ("electronics", "kitchen"),
                          ("kitchen", "books"), ("kitchen", "dvd"), ("kitchen", "electronics")]
        self.num_classes = 2
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 5000
        self.kernel_size = 10
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 128
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        
        self.model = Box({
            "n_hidden": 128
        })
                
        self.dataloader = Box({
            "reset_and_reload_memory": False,
            "dataset": "AmazonReviews",
            "AmazonReviews": {
                "filename":"amazon.mat",
                "n_features": 5000,
                "num_workers": 8,
                "normalize": True,
                "domains": ["books", "dvd", "electronics", "kitchen"],
                "data_root": "data/amazon_reviews"
            }
        })


class TRANSFORMED_MOONS(object):
    def __init__(self):
        super(TRANSFORMED_MOONS, self).__init__()        
        self.class_names = ['0', '1']
        self.sequence_len = 128
        self.scenarios = [("0", "1")]
        self.num_classes = 2
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 2
        self.kernel_size = 1
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 128
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 128
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        
        self.model = Box({
            "n_hidden": 128
        })
                
        self.dataloader = Box({
            "dataset": "MoonsNS",
            "MoonsNS": {
                "source_train_x": "data/moons/moons_source_train_x.npy",
                "source_train_y": "data/moons/moons_source_train_y.npy",
                "target_train_x": "data/moons/moons_target_train_x.npy",
                "target_train_y": "data/moons/moons_target_train_y.npy",

                "source_valid_x": "data/moons/moons_source_valid_x.npy",
                "source_valid_y": "data/moons/moons_source_valid_y.npy",
                "target_valid_x": "data/moons/moons_target_valid_x.npy",
                "target_valid_y": "data/moons/moons_target_valid_y.npy",

                "source_test_x": "data/moons/moons_source_test_x.npy",
                "source_test_y": "data/moons/moons_source_test_y.npy",
                "target_test_x": "data/moons/moons_target_test_x.npy",
                "target_test_y": "data/moons/moons_target_test_y.npy"
            }
        })
