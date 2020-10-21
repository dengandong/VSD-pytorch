import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, images, names, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class)  -- the HTML webpage class that stores these imaegs (see html.py for more details)
        images (numpy array list) -- a list of numpy array that stores images
        names (str list)          -- a str list stores the names of the images above
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    name = ntpath.basename(image_path)

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in zip(names, images):
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = True
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, display_id_plus=50, title='frames', preproc_mode=1,
                                separate=True):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
            display_id_plus - - define in vis(win=)
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if not separate:  # show all the images in one visdom panel
                # ncols = min(ncols, len(visuals))
                try:
                    h, w = next(iter(visuals.values())).shape[3:]  # my # B x T x C x H x W
                except ValueError:
                    h, w = next(iter(visuals.values())).shape[2:]  # my # B x C x H x W
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                # title = self.name
                label_html = ''
                label_html_row = ''
                frames = []
                idx = 0
                for label, frame in visuals.items():
                    frames_numpy = util.tensor2frames(frame[0],
                                                      preprocess_mode=preproc_mode)  # my # 1st video in the batch of 32, T x C x H x W
                    # print(label, np.shape(frames_numpy))
                    label_html_row += '<td>%s</td>' % label
                    frames.append(frames_numpy)  # my # multi-image
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                frames = np.concatenate(frames, 0)  # my # 48 x c x h x w
                # white_image = np.ones([h, w, 3]) * 255
                # while idx % ncols != 0:
                #     frames.append(white_image)
                #     label_html_row += '<td></td>'
                #     idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(frames, nrow=ncols, win=self.display_id + display_id_plus,
                                    padding=2, opts=dict(title=title + ' in epoch {}'.format(epoch)))
                    # label_html = '<table>%s</table>' % label_html
                    # self.vis.text(table_css + label_html, win=self.display_id + display_id_plus + 1,
                    #               opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # show each image in separate visdom panels;
                idx = 1
                try:
                    for label, frames in visuals.items():
                        image_numpy = util.tensor2frames(frames[0])  # if not denorm ??
                        self.vis.images(image_numpy, nrow=ncols, opts=dict(title=label),
                                        win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # # save images to the disk
            # for label, image in visuals.items():
            #     image_numpy = util.tensor2im(image)
            #     img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            #     util.save_image(image_numpy, img_path)

            # # update website
            # webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            # for n in range(epoch, 0, -1):
            #     webpage.add_header('epoch [%d]' % n)
            #     ims, txts, links = [], [], []
            #
            #     for label, image_numpy in visuals.items():
            #         image_numpy = util.tensor2im(image)
            #         img_path = 'epoch%.3d_%s.png' % (n, label)
            #         ims.append(img_path)
            #         txts.append(label)
            #         links.append(img_path)
            #     webpage.add_images(ims, txts, links, width=self.win_size)
            # webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_variable_norm(self, epoch, counter_ratio, variables, idx=100, mode='l2'):
        """display the current norm of variables on visdom display
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            variables (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_norm'):
            self.plot_norm = {'X': [], 'Y': [], 'legend': list(variables.keys())}
        self.plot_norm['X'].append(epoch + counter_ratio)

        if mode == 'l1':
            self.plot_norm['Y'].append(
                [np.linalg.norm(variables[k].cpu().detach().numpy(), ord=1)  # / variables[k].numel()
                 for k in self.plot_norm['legend']])
            pass
        elif mode == 'l2':
            self.plot_norm['Y'].append([np.linalg.norm(variables[k].cpu().detach().numpy())  # / variables[k].numel()
                                        for k in self.plot_norm['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_norm['X'])] * len(self.plot_norm['legend']), 1),
                Y=np.array(self.plot_norm['Y']),
                opts={
                    'title': self.name + mode + ' norm of the frames',
                    'legend': self.plot_norm['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'norm'},
                win=self.display_id + idx)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_layers_norm(self, epoch, counter_ratio, variables, idx=90, mode='l2'):
        """display the current norm of variables on visdom display
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            variables (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_norm_layer'):
            self.plot_norm_layer = {'X': [], 'Y': [], 'legend': list(variables.keys())}
        self.plot_norm_layer['X'].append(epoch + counter_ratio)

        if mode == 'l1':
            self.plot_norm_layer['Y'].append(
                [np.linalg.norm(variables[k].cpu().detach().numpy(), ord=1)  # / variables[k].numel()
                 for k in self.plot_norm_layer['legend']])
            pass
        elif mode == 'l2':
            self.plot_norm_layer['Y'].append([np.linalg.norm(variables[k].cpu().detach().numpy())  # / variables[k].numel()
                                        for k in self.plot_norm_layer['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_norm_layer['X'])] * len(self.plot_norm_layer['legend']), 1),
                Y=np.array(self.plot_norm_layer['Y']),
                opts={
                    'title': self.name + mode + ' norm of each layer in G',
                    'legend': self.plot_norm_layer['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'norm'},
                win=self.display_id + idx)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
