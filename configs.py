class Config:
    def __init__(self):
        self.device_ids = [0, 1, 2, 3]
        self.epoch_num = 60  # Number of epochs for training
        self.warmup_num = 4
        self.resume_epoch_num = 0  # Default is 0, change if want to resume
        self.test_freq = 1  # Run on test set every nTestInterval epochs
        self.save_freq = 10  # Store a model every snapshot epochs
        self.lr = 4e-2
        self.multiplier = 20
        self.clip_len = 32
        self.num_workers = 4
        self.batch_size = 16
        self.weight_decay = 1e-4
        self.milestones = [25, 35, 45]
        self.freeze_bn = False
        self.pretrain = False
        self.use_sim = False
        self.use_test = True  # See evolution of the test set when training
        self.set_optim = 'SGD'
        self.dataset = 'something'  # Options: hmdb51 or ucf101
        self.classes_num = self.get_classes_num(self.dataset)
        self.model_name = 'dnet50'
        self.resume_model_path = None
        if self.pretrain:
            self.pretrain_path = self.get_pretrained_model(self.model_name)
        else:
            self.pretrain_path = None

    def get_classes_num(self, dataset):
        if dataset == 'hmdb51':
            return 51
        elif dataset == 'ucf101':
            return 101
        elif dataset == 'something':
            return 10  # 174
        else:
            print('We only implemented hmdb and ucf datasets.')
            raise NotImplementedError

    def get_pretrained_model(self, model_name):
        if model_name == 'I3D':
            return 'network/rgb_imagenet.pkl'
        elif model_name == 'R3D':
            return 'network/resnet-18-kinetics.pth'
        else:
            print('We only implemented hmdb and ucf datasets.')
            raise NotImplementedError


class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '../data/ucf101'

            # Save preprocess data into output_dir
            output_dir = '../data/ucf101_pdata'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '../data/hmdb'

            output_dir = '../data/hmdb_pdata/'

            return root_dir, output_dir
        elif database == 'something':
            # folder that contains class labels
            root_dir = '../data/something'

            output_dir = '../data/something_pdata/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'models/c3d-pretrained.pth'
