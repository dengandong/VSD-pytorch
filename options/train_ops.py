import argparse


class TrainOps():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # parser = argparse.ArgumentParser(description='Video Self-disentanglement, PyTorch Version')
        # basic config
        parser.add_argument('--frames_saved_path', type=str, default='data/101frames_16')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--split_type', type=str, default='01')

        parser.add_argument('--model_save_path', type=str, default='saved_model')
        parser.add_argument('--model_save_date', type=str, required=True)  # e.g. 'July29'
        parser.add_argument('--is_variation', type=bool, default=False)
        parser.add_argument('--net_type', type=str, default='gru', help='gru or 3d')

        # training config
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--distributed', type=bool, default=True)
        parser.add_argument('--optimizer_mode', type=str, default='Adam')
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--grad_accumulation', type=str, default=False)
        parser.add_argument('--lr_reduce_mode', type=str, default='step', help='step or adaptive')
        parser.add_argument('--lr_reduce_epoch', type=int, default=10, help='if args.ReduceLROnPlateau, this mean patience')
        parser.add_argument('--lr_reduce_ratio', type=float, default=0.1)
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--test_interval', type=int, default=20)
        parser.add_argument('--save_model_interval', type=int, default=1)

        args = parser.parse_args()

        print(str(args), '\n')

        self.initialized = True
