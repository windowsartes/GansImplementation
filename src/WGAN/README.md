# WGAN implementation
## OVerview
This implementation is totally based on [this](https://arxiv.org/abs/1701.07875) paper. Since it's so hard to understand, you can read [this](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html) awesome post - it's so helpful.

I've used the same architecture that [DCGAN](https://arxiv.org/abs/1511.06434) paper proposed. But there are some changes. First of all, now it's called Critic. Also there is no last activation layer for now, cause Critic's output can be every real number, not only [0,1].

I don't have enough resources to train it for a long time, so I've trained it during 10 eepochs just to to see if model trains, whether it progresses or not. 

[There](https://www.kaggle.com/code/nikolaimakarov/wgan-implementation-and-training/script) you can find my kaggle notebook with training and wandb logging. At the wandb run you can also find loss graphs and generated images after each epoch. As you may notice, for sure model progresses, so my implementation is correct.