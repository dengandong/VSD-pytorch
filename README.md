# video self-disentanglement

This is a pytorch implementaion of video disentanglement in a self-supervised manner.

The framework aims at disentangling the motion and content of a video in an unsupervised way; after that, use the extracted motion code to generate frames with similiar motion and different content, we also use the same motion encoder after the generator to extract the motion code of the synthesized frames and expect the motion code to keep unchaged in this generation process, i.e. view the motion as a supervision.

We expect this model to extract reasonable motion representation to facilitate action recognition and frame synthesis.

![Deterministic Autoencoder](http://gitlab.sz.sensetime.com/dengandong/video-self-disentanglement/blob/master/images/Fig_1.jpg)
![Flipped Autoencoder](http://gitlab.sz.sensetime.com/dengandong/video-self-disentanglement/blob/master/images/Fig_2.jpg)


model ---> model/network.py

motion encoder ---> model/ae_gru.py

content encoder ---> model/content_encoder.py
